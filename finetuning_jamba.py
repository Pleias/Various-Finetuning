from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments,  BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int4_skip_modules=["mamba"]
)

tokenizer = AutoTokenizer.from_pretrained("jamba")
model = AutoModelForCausalLM.from_pretrained(
    "jamba", trust_remote_code=True, 
    device_map='auto', 
    attn_implementation="flash_attention_2", 
    quantization_config=quantization_config, 
    use_mamba_kernels=False #Disabling the mamba kernels since I have a recurrent error.
    )

dataset = load_dataset("Abirate/english_quotes", split="train")
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-3
)
lora_config = LoraConfig(
    r=8,
    target_modules=["embed_tokens", "x_proj", "in_proj", "out_proj"],
    task_type="CAUSAL_LM",
    bias="none"
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    dataset_text_field="quote",
)

trainer.train()