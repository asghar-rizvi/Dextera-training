import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# 1. MODEL CONFIGURATION
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# 2. LEGAL SYSTEM PROMPT
NEW_SYSTEM_PROMPT = """Role: Specialized Expert in Pakistani Criminal Law (PPC & CrPC).
Rules of Engagement:
1. SCOPE: You only answer queries related to Pakistani criminal statutes, precedents, and procedures.
2. REFUSAL: For ANY non-legal or non-Pakistani criminal law question, you MUST respond with: "I'm designed to assist only with questions related to Pakistani criminal law. Please ask a question about criminal law in Pakistan."
3. CITATIONS: Always cite the relevant Section of the Pakistan Penal Code (PPC) or Article of the Code of Criminal Procedure (CrPC) when providing legal information.
4. TONE: Maintain a formal, objective, and analytical tone.
5. CONTEXT: If legal text is provided in the query, prioritize that text for your reasoning.

Constraint: Do not provide personal opinions or general legal advice.
Reminder: Act Normal when greet and If i asked anything other than pakistan criminal law just SAY: 'NOT A PAKISTAN CRIMINAL LAW QUERY '"""

# 3. LOAD TOKENIZER
print(f"Loading tokenizer: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

# 4. LOAD MODEL
# Changed to 'dtype' to fix deprecated warning
print(f"Loading BF16 Model: {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16, 
    device_map="auto",
)

# 5. PEFT (LoRA) SETUP
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 6. DATASET PREPARATION
def format_prompt(examples):
    texts = []
    for q, a in zip(examples["question"], examples["answer"]):
        messages = [
            {"role": "system", "content": NEW_SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ]
        formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(formatted_chat)
        
    return {"text": texts}

print("Loading dataset...")
dataset = load_dataset("data", split="train") 
dataset = dataset.map(format_prompt, batched=True)

# 7. TRAINING CONFIGURATION (Memory Optimized)
training_args = SFTConfig(
    output_dir="./llama_pak_legal_v1",
    max_length=2048,
    per_device_train_batch_size=2,        # Reduced from 4 to save VRAM
    gradient_accumulation_steps=8,     # Increased to keep effective batch size = 16
    learning_rate=2e-5,
    bf16=True,
    logging_steps=10,
    num_train_epochs=1,
    save_steps=100,
    optim="paged_adamw_8bit",          # FIX: Use 8-bit optimizer to save ~8GB VRAM
    dataset_text_field="text",
    packing=False,                     # Disabled for better memory stability
    gradient_checkpointing=True,
    report_to="none"
)

# 8. TRAIN
print("Starting training on L4 GPU...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# 9. SAVE
model.save_pretrained("pak_legal_llama_adapter")
tokenizer.save_pretrained("pak_legal_llama_adapter")
print("Training Complete! Adapter saved to 'pak_legal_llama_adapter'")