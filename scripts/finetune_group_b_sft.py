import os
import torch
import json
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

# Configurations
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "./outputs/mistral-7b-group-b-sft"
DATASET_PATH_STAGE1 = "./data/group_b_stage1.json"
DATASET_PATH_STAGE2 = "./data/group_b_stage2.json"

def main():
    print("Initializing Unsloth with Mistral-7B-Instruct-v0.3...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=8192, # Long context for RAG context
        dtype=None,
        load_in_4bit=True,
    )

    # Configure LoRA adapters - higher rank for reasoning
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

    # SMILES specific tokenizer additions
    # Let's add SMILES-specific tokens to prevent bad token splits
    smiles_tokens = ["[NH]", "[NH2]", "[NH3+]", "[OH]", "[CH2]", "C#N", "C=O", "S=O", "P=O"]
    num_added_toks = tokenizer.add_tokens(smiles_tokens)
    print(f"Added {num_added_toks} SMILES tokens to tokenizer.")
    model.resize_token_embeddings(len(tokenizer))

    # Prompts for MolRAG reasoning
    molrag_prompt = """<s>[INST] You are a toxicology reasoning system (MolRAG).
Analyze the following molecule SMILES: {smiles}
Baseline verdict: {baseline}
Retrieved knowledge contexts: {contexts}
Provide a complete reasoning mechanism and return in strict JSON schema. [/INST]
{response}</s>"""

    def format_prompts(batch):
        texts = []
        for smiles, baseline, contexts, response in zip(
            batch["smiles"], batch["baseline"], batch["contexts"], batch["response"]
        ):
            text = molrag_prompt.format(
                smiles=smiles,
                baseline=json.dumps(baseline),
                contexts=json.dumps(contexts),
                response=json.dumps(response)
            )
            texts.append(text)
        return {"text": texts}

    # Helper to load/create dataset
    def get_dataset(path, name):
        if os.path.exists(path):
            return load_dataset("json", data_files=path, split="train")
        else:
            print(f"{name} dataset not found, generating dummy template...")
            dummy_data = [
                {
                    "smiles": "CCO",
                    "baseline": {"label": "Non-toxic", "score": 0.1},
                    "contexts": ["Ethanol is metabolized by alcohol dehydrogenase to acetaldehyde."],
                    "response": {
                        "evidence_overview": "Ethanol is a small molecular weight solvent.",
                        "longform_summary": "Ethanol is non-toxic at screening levels.",
                        "mechanism_chain": ["Ethanol -> Alcohol Dehydrogenase -> Acetaldehyde"],
                        "key_substructures": ["Hydroxyl group"],
                        "confidence_rationale": "High similarity and clear metabolic path.",
                        "suggested_label": "Non-toxic",
                        "confidence": 0.95
                    }
                }
            ]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(dummy_data, f)
            return load_dataset("json", data_files=path, split="train")

    # Curriculum training - Stage 1 (Mechanism Chain)
    print("--- Stage 1: SFT (Mechanism Chain Focus) ---")
    dataset_s1 = get_dataset(DATASET_PATH_STAGE1, "Stage 1").map(format_prompts, batched=True)

    training_args_s1 = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        num_train_epochs=3,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=f"{OUTPUT_DIR}_stage1",
    )

    trainer_s1 = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_s1,
        dataset_text_field="text",
        max_seq_length=8192,
        dataset_num_proc=2,
        args=training_args_s1,
    )
    trainer_s1.train()

    # Curriculum training - Stage 2 (Full JSON Response Schema Compliance)
    print("--- Stage 2: SFT (Full JSON Schema Compliance) ---")
    dataset_s2 = get_dataset(DATASET_PATH_STAGE2, "Stage 2").map(format_prompts, batched=True)

    training_args_s2 = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        num_train_epochs=2,
        learning_rate=5e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=OUTPUT_DIR,
    )

    trainer_s2 = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_s2,
        dataset_text_field="text",
        max_seq_length=8192,
        dataset_num_proc=2,
        args=training_args_s2,
    )
    trainer_s2.train()

    print("Saving fine-tuned LoRA adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Exporting GGUF
    print("Exporting model to GGUF format...")
    model.save_pretrained_gguf(
        f"{OUTPUT_DIR}-gguf",
        tokenizer,
        quantization_method="q4_k_m"
    )
    print("Training and GGUF export complete.")

if __name__ == "__main__":
    main()
