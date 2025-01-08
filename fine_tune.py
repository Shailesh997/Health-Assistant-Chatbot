import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load the dataset (you can replace this with your dataset)
train_data = []
valid_data = []

# Load your train and valid datasets (loading only the first 100 lines for debugging)
with open("train.jsonl", "r") as f:
        for _ in range(100):  # Load only the first 100 entries for debugging
            train_data.append(json.loads(f.readline()))

with open("valid.jsonl", "r") as f:
        for _ in range(100):  # Load only the first 100 entries for debugging
            valid_data.append(json.loads(f.readline()))
# Convert data to HuggingFace Dataset format
train_dataset = Dataset.from_dict({"input": [entry['input'] for entry in train_data], 
                                       "output": [entry['output'] for entry in train_data]})
    
valid_dataset = Dataset.from_dict({"input": [entry['input'] for entry in valid_data], 
                                       "output": [entry['output'] for entry in valid_data]})

# Use a smaller model for debugging (e.g., gpt-2)
model_name = "gpt2"  # Switching to gpt2 for debugging
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(examples):
    # Concatenate the input situation and the emotion in a format that the model can use to generate a response
    inputs = examples['input']
    targets = examples['output']

    # Create a string where input is followed by the emotion (as target)
    inputs_and_labels = [f"{input_text} Emotion: {target}" for input_text, target in zip(inputs, targets)]

    # Tokenize the combined text (situation + emotion)
    encodings = tokenizer(inputs_and_labels, padding="max_length", truncation=True, max_length=128)

    # Set the labels for training
    encodings["labels"] = encodings["input_ids"].copy()  # For causal LM, labels are the same as input_ids
    return encodings


# Tokenizing the dataset
train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)

# Training Arguments for fine-tuning
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save the model outputs
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    learning_rate=2e-5,  # Learning rate for fine-tuning
    per_device_train_batch_size=2,  # Batch size for training
    per_device_eval_batch_size=2,   # Batch size for evaluation
    num_train_epochs=3,  # Number of epochs to train
    weight_decay=0.01,  # Weight decay for regularization
    logging_dir="./logs",  # Directory for logs
    logging_steps=10,  # Log every 10 steps
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_model")  # Save the model to the specified directory
tokenizer.save_pretrained("./fine_tuned_model")  # Save the tokenizer

# Check if the model is saved successfully
import os
if os.path.exists("./fine_tuned_model"):
    print("Model saved successfully!")
else:
    print("Model saving failed.")
