import os
import json  
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

# Configurations
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./outputs/qwen2.5-7b-group-a-sft"
DATASET_PATH = "./data/group_a_tool_calling.json" # local compiled tool-calling dataset

def main():
    print("Initializing Unsloth with Qwen 2.5-7B-Instruct...")
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=4096,
        dtype=None, # None for auto detection (Float16/Bfloat16)
        load_in_4bit=True, # Use 4bit quantization to save VRAM
    )

    # Configure LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth", # Saves VRAM
        random_state=3407,
    )

    # Prompt template for tool calling / validation
    tool_calling_prompt = """<|im_start|>system
You are a tool calling assistant. Below is a set of tools you can use.
Tools: {tools}
Respond in strict JSON tool calling format.<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""

    def format_prompts(batch):
        texts = []
        for tools, query, response in zip(batch["tools"], batch["query"], batch["response"]):
            text = tool_calling_prompt.format(
                tools=json.dumps(tools),
                query=query,
                response=json.dumps(response)
            ) + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    # Load dataset
    if os.path.exists(DATASET_PATH):
        print(f"Loading dataset from {DATASET_PATH}...")
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    else:
        print("Dataset not found. Using a representative dummy dataset for structure initialization.")
        # Create a mock training set
        dummy_data = [
            {
                "tools": [
                    {
                        "name": "validate_smiles",
                        "description": "Verify molecular SMILES validity.",
                        "parameters": {"type": "object", "properties": {"smiles": {"type": "string"}}}
                    }
                ],
                "query": "Please validate SMILES CCO",
                "response": {"name": "validate_smiles", "arguments": {"smiles": "CCO"}}
            }
        ]
        # Write dummy to run/verify code shape
        os.makedirs("./data", exist_ok=True)
        with open(DATASET_PATH, "w") as f:
            json.dump(dummy_data, f)
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    dataset = dataset.map(format_prompts, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=OUTPUT_DIR,
    )

    print("Starting fine-tuning...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=4096,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    trainer.train()

    print("Saving fine-tuned LoRA adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Exporting GGUF for vLLM serving
    print("Exporting model to GGUF format...")
    # Unsloth supports direct saving to GGUF format (Q4_K_M)
    model.save_pretrained_gguf(
        f"{OUTPUT_DIR}-gguf",
        tokenizer,
        quantization_method="q4_k_m"
    )
    print("Training and GGUF export complete.")

if __name__ == "__main__":
    main()
