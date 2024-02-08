import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, GenerationConfig, pipeline, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset, DatasetDict
import huggingface_hub

# model_name = "mistralai/Mistral-7B-v0.1"
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
dataset_name = "akemiH/NoteChat"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

raw_dataset = load_dataset(dataset_name)
train_test = raw_dataset["train"].train_test_split(test_size=0.1)
test_valid = train_test['test'].train_test_split(test_size=0.5)
dataset = DatasetDict({
    'train': train_test['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

context_length = 1024

def tokenize_dataset(element):
  outputs = tokenizer(element["data"], truncation=True, max_length = context_length, return_overflowing_tokens=True, return_length=True)
  return outputs

tokenized_dataset = dataset.map(tokenize_dataset, batched=True, remove_columns=dataset["train"].column_names)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

huggingface_hub.login(token="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")

training_args = TrainingArguments(
    output_dir="mistral-continual-pretraining-notechat",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

trainer.train()