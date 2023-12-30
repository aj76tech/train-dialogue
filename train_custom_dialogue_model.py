from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch

# Load DialoGPT-medium model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load your custom dataset (replace 'path_to_your_dataset.txt' with your actual dataset path)
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='path_to_your_dataset.txt',
    block_size=128  # Adjust the block size according to your dataset
)

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir="./shopkeepr_model",  # Set the output directory
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust the number of training epochs
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
