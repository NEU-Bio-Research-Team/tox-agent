import os
import torch
import json
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel
from grpo_rewards import (
    toxicity_label_reward,
    json_schema_reward,
    mechanism_chain_quality,
    confidence_calibration
)

# Configurations
SFT_CHECKPOINT_DIR = "./outputs/mistral-7b-group-b-sft"
OUTPUT_DIR = "./outputs/mistral-7b-group-b-grpo"
DATASET_PATH = "./data/group_b_grpo.json"

def main():
    print(f"Initializing Unsloth with SFT base model {SFT_CHECKPOINT_DIR}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_CHECKPOINT_DIR,
        max_seq_length=8192,
        dtype=None,
        load_in_4bit=True,
    )

    # Configure LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Prompts for GRPO (excludes response since the model generates trajectories)
    molrag_grpo_prompt = """<s>[INST] You are a toxicology reasoning system (MolRAG).
Analyze the following molecule SMILES: {smiles}
Baseline verdict: {baseline}
Retrieved knowledge contexts: {contexts}
Provide a complete reasoning mechanism and return in strict JSON schema. [/INST]
"""

    def format_prompts(batch):
        prompts = []
        for smiles, baseline, contexts in zip(batch["smiles"], batch["baseline"], batch["contexts"]):
            prompt = molrag_grpo_prompt.format(
                smiles=smiles,
                baseline=json.dumps(baseline),
                contexts=json.dumps(contexts)
            )
            prompts.append(prompt)
        return {"prompt": prompts}

    # Load dataset
    if os.path.exists(DATASET_PATH):
        print(f"Loading GRPO dataset from {DATASET_PATH}...")
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    else:
        print("GRPO dataset not found, generating dummy template...")
        dummy_data = [
            {
                "smiles": "CCO",
                "baseline": {"label": "Non-toxic", "score": 0.1},
                "contexts": ["Ethanol is metabolized by alcohol dehydrogenase to acetaldehyde."],
                "label_targets": "non-toxic",
                "max_similarities": 0.95
            }
        ]
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        with open(DATASET_PATH, "w") as f:
            json.dump(dummy_data, f)
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    dataset = dataset.map(format_prompts, batched=True)

    # GRPO Training Configurations
    training_args = GRPOConfig(
        use_vllm=True, # Set to True if vLLM serving is active during training
        vllm_device="cuda:0",
        vllm_gpu_memory_utilization=0.4, # Leave VRAM budget for model training
        learning_rate=2e-6, # Lower LR for RL fine-tuning
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_prompt_length=4096,
        max_completion_length=4096,
        num_generations=4, # Group size G (number of generated trajectories per prompt)
        output_dir=OUTPUT_DIR,
    )

    print("Initializing GRPOTrainer...")
    # Reward functions list
    reward_funcs = [
        toxicity_label_reward,
        json_schema_reward,
        mechanism_chain_quality,
        confidence_calibration
    ]

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting GRPO training loop...")
    trainer.train()

    print("Saving RL fine-tuned adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Exporting GGUF
    print("Exporting RL model to GGUF format...")
    model.save_pretrained_gguf(
        f"{OUTPUT_DIR}-gguf",
        tokenizer,
        quantization_method="q4_k_m"
    )
    print("GRPO training and GGUF export complete.")

if __name__ == "__main__":
    main()
