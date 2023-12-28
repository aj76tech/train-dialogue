# train_custom_dialogue_model.py

import json
import numpy as np
import gradio as gr
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer

# Step 1: Define a Gradio Interface to Upload Dataset
def upload_dataset(file):
    with open(file.name, 'r') as f:
        dataset = json.load(f)
    return dataset

iface = gr.Interface(
    fn=upload_dataset,
    inputs=gr.File(label="Upload dataset.json"),
    outputs="json",
    live=True,
    share=True
)

# Display the Gradio Interface to upload the dataset
iface.launch()
share_url = iface.share()
print("Shareable link:", share_url)
# Wait for the user to upload the dataset and continue to the next cell

# Step 2: Load and Preprocess the Uploaded Dataset
# The uploaded dataset will be available in the variable iface.outputs
dataset = iface.outputs

# Preprocess the dataset
def preprocess_data(example):
    # Your custom preprocessing logic here
    return example

dataset = [preprocess_data(example) for example in dataset]

# Step 3: Initialize Tokenizer and Model
tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-medium')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('microsoft/DialoGPT-medium')

# Step 4: Encode Your Dataset
def encode_data(example):
    encoded = tokenizer(example['dialog'], truncation=True, padding='max_length', max_length=128)
    encoded['labels'] = encoded['input_ids'][:]
    return encoded

encoded_dataset = [encode_data(example) for example in dataset]

# Step 5: Set Up Training
output_dir = "your_output_directory"  # Specify the output directory
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=4,  # Adjust as needed
    per_device_eval_batch_size=4,   # Adjust as needed
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=None,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    eval_dataset=encoded_dataset[:10]  # Select the first 10 examples for evaluation
)

# Step 6: Fine-Tune the Model
trainer.train()

# Step 7: Save the Trained Model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Step 8: Evaluate and Generate Predictions
post_eval_results = trainer.evaluate(encoded_dataset[:10])  # Select the first 10 examples for evaluation
post_val_predictions = trainer.predict(encoded_dataset[:10])  # Select the first 10 examples

print('Evaluation Results after fine-tuning:', post_eval_results['eval_loss'])

# Step 9: Explore Predictions
for idx, (example, post) in enumerate(zip(dataset[:10], post_val_predictions.predictions)):
    post_pred = tokenizer.decode(np.argmax(post, axis=-1), skip_special_tokens=True)
    ground_truth = example["dialog"]
    
    print('Ground truth \n' + ground_truth + '\n')
    print('Post-prediction \n' + post_pred + '\n')
    print('----------------------------------------------------------------------------------------------------------------------\n')
