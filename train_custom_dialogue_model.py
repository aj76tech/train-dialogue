from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
import requests
import os
from IPython.display import HTML
from google.colab import files

# Load DialoGPT-medium model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load your custom dataset
github_dataset_url = 'https://raw.githubusercontent.com/aj76tech/train-dialogue/main/dataset.txt'
local_file_path = 'dataset.txt'

# Download the file
response = requests.get(github_dataset_url)
with open(local_file_path, 'w', encoding='utf-8') as file:
    file.write(response.text)

# Use the local file path in the TextDataset constructor
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=local_file_path,
    block_size=128,
    overwrite_cache=True,
    cache_dir='./cache'
)

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir="./shopkeepr_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./shopkeepr_model")
tokenizer.save_pretrained("./shopkeepr_model")
