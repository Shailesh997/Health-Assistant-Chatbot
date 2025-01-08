import json
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("bdotloh/empathetic-dialogues-contexts")

# Function to preprocess data
def preprocess_data(dataset):
    formatted_data = []
    for entry in dataset:
        input_text = f"Situation: {entry['situation']}\nEmotion: {entry['emotion']}\nResponse:"
        output_text = ""  # Leave blank for supervised fine-tuning
        formatted_data.append({"input": input_text, "output": output_text})
    return formatted_data

# Apply preprocessing to train and validation sets
train_data = preprocess_data(dataset["train"])
valid_data = preprocess_data(dataset["validation"])

# Save as JSONL files
with open("train.jsonl", "w") as train_file:
    for entry in train_data:
        train_file.write(json.dumps(entry) + "\n")

with open("valid.jsonl", "w") as valid_file:
    for entry in valid_data:
        valid_file.write(json.dumps(entry) + "\n")
