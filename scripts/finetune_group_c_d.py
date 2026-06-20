import os
import torch
import json
from datasets import load_dataset
from trl import ORPOTrainer, ORPOConfig
from transformers import AutoTokenizer, TrainingArguments
from unsloth import FastLanguageModel

# Configurations
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./outputs/qwen2.5-7b-group-c-orpo"
DATASET_PATH = "./data/group_c_writer_preference.json"

def main():
    print("Initializing Unsloth for Writer Agent (Group C) ORPO Preference Training...")
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=8192, # Writer agent needs long context (8k)
        dtype=None,
        load_in_4bit=True,
    )

    # Configure LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Prompts for ORPO (Writer Agent report generation)
    writer_prompt = """<|im_start|>system
You are a pharmacology and toxicology report synthesis writer. Write a detailed final toxicology report based on screening and research results. Respond in strict JSON.<|im_end|>
<|im_start|>user
Screening: {screening}
Research: {research}
Language: {language}<|im_end|>
<|im_start|>assistant
"""

    def format_prompts(batch):
        prompts = []
        chosens = []
        rejecteds = []
        for screening, research, language, chosen, rejected in zip(
            batch["screening"], batch["research"], batch["language"], batch["chosen"], batch["rejected"]
        ):
            prompt = writer_prompt.format(
                screening=json.dumps(screening),
                research=json.dumps(research),
                language=language
            )
            prompts.append(prompt)
            chosens.append(json.dumps(chosen) + tokenizer.eos_token)
            rejecteds.append(json.dumps(rejected) + tokenizer.eos_token)
            
        return {
            "prompt": prompts,
            "chosen": chosens,
            "rejected": rejecteds
        }

    # Load dataset
    if os.path.exists(DATASET_PATH):
        print(f"Loading preference dataset from {DATASET_PATH}...")
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    else:
        print("Preference dataset not found, generating dummy template...")
        dummy_data = [
            {
                "screening": {"summary": "Clinical=TOXIC, assay_hits=3", "final_verdict": "TOXIC"},
                "research": {"consensus_mechanisms": ["hERG blocking"]},
                "language": "vi",
                "chosen": {
                    "report_metadata": {"smiles": "CCO", "language": "vi"},
                    "executive_summary": "Hợp chất CCO có nguy cơ độc tính lâm sàng cao.",
                    "risk_level": "HIGH",
                    "sections": {"clinical_toxicity": {}, "mechanism_toxicity": {}}
                },
                "rejected": {
                    "report_metadata": {"smiles": "CCO", "language": "vi"},
                    "executive_summary": "Không có dữ liệu gì.",
                    "risk_level": "UNKNOWN",
                    "sections": {}
                }
            }
        ]
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        with open(DATASET_PATH, "w") as f:
            json.dump(dummy_data, f)
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    dataset = dataset.map(format_prompts, batched=True)

    # ORPO Training arguments
    orpo_args = ORPOConfig(
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        max_prompt_length=4096,
        max_length=8192,
        beta=0.1, # Weight for ORPO loss
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        optim="adamw_8bit",
        logging_steps=1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        output_dir=OUTPUT_DIR,
    )

    print("Initializing ORPOTrainer...")
    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting ORPO training loop...")
    trainer.train()

    print("Saving fine-tuned adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Exporting GGUF
    print("Exporting Writer model to GGUF format...")
    model.save_pretrained_gguf(
        f"{OUTPUT_DIR}-gguf",
        tokenizer,
        quantization_method="q4_k_m"
    )
    print("ORPO training and GGUF export complete.")

if __name__ == "__main__":
    main()
