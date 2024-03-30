from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments,  BitsAndBytesConfig

#Check if you do not have any import issue to use the Fast Mamba Kernel
#Will (very appropriately) break before loading the weights.
import mamba_ssm

#With 4bit quants have to manually correct modeling_jamba.py on l. 1070:
#if not is_fast_path_available or "cuda" not in self.x_proj.weight.device.type:
#becoming:
#if not is_fast_path_available:

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int4_skip_modules=["mamba"] #Maybe not necessary (per axoltl) but to test.
)

tokenizer = AutoTokenizer.from_pretrained("jamba")

dataset = load_dataset("Abirate/english_quotes", split="train")
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim = "adamw_8bit",
    max_grad_norm = 0.3,
    weight_decay = 0.001,
    warmup_ratio = 0.03,
    gradient_checkpointing=True,
    logging_dir='./logs',
    logging_steps=1,
    max_steps=50,
    group_by_length=True,
    lr_scheduler_type = "linear",
    learning_rate=2e-3
)
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    init_lora_weights=False,
    r=8,
    target_modules=["embed_tokens", "x_proj", "in_proj", "out_proj"],
    task_type="CAUSAL_LM",
    bias="none"
)

model = AutoModelForCausalLM.from_pretrained(
    "jamba",
    trust_remote_code=True, 
    device_map='auto',
    attn_implementation="flash_attention_2", 
    quantization_config=quantization_config, 
    use_mamba_kernels=True
    )

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    max_seq_length = 256,
    dataset_text_field="quote",
)

trainer.train()
