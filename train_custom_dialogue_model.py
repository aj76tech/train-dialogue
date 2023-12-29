import os
import numpy as np
import tempfile
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from datasets import load_dataset

# Step 1: Load and Preprocess Your Custom Dataset
dataset_url = "https://raw.githubusercontent.com/aj76tech/train-dialogue/main/dataset.json"
dataset = load_dataset('json', data_files=dataset_url)

# Preprocess the dataset
def preprocess_data(example):
    # Your custom preprocessing logic here
    return example

# Apply the function to all examples in the dataset
dataset = dataset.map(preprocess_data)

# Step 2: Initialize Tokenizer and Model
tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-medium')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('microsoft/DialoGPT-medium')

# Step 3: Encode Your Dataset
def encode_data(examples):
    # Handle single sequences or pairs of sequences
    if "dialog" in examples:
        # Single sequence
        return tokenizer(examples["dialog"], truncation=True, padding="max_length", max_length=128)
    elif "dialogue_1" in examples and "dialogue_2" in examples:
        # Pairs of sequences
        return tokenizer(examples["dialogue_1"], examples["dialogue_2"], truncation=True, padding="max_length", max_length=128)
    else:
        raise ValueError("Invalid input format. Please provide 'dialog' or ('dialogue_1', 'dialogue_2') in examples.")

# Apply the function to all examples in the dataset
encoded_dataset = dataset.map(encode_data, batched=True)

# Step 4: Set Up Training
output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "saved_model")
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,            # specify the output directory
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=None
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation']
)

# Step 5: Fine-Tune the Model
trainer.train()

# Step 6: Save the Trained Model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Step 7: Evaluate and Generate Predictions
post_eval_results = trainer.evaluate(encoded_dataset['validation'])
post_val_predictions = trainer.predict(encoded_dataset['validation'].select(range(10)))

print('Evaluation Results after fine-tuning:', post_eval_results['eval_loss'])

# Step 8: Explore Predictions
for idx, (_, post) in enumerate(zip(encoded_dataset['validation'][:10], post_val_predictions.predictions)):
    post_pred = tokenizer.decode(np.argmax(post, axis=-1), skip_special_tokens=True)
    ground_truth = encoded_dataset['validation'][idx]["dialog"]
    
    print('Ground truth \n' + ground_truth + '\n')
    print('Post-prediction \n' + post_pred + '\n')
    print('----------------------------------------------------------------------------------------------------------------------\n')
