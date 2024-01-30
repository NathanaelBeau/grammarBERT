import json
import gzip
from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from transformers import EvalPrediction

from transformers import RobertaForMaskedLM, RobertaTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer, TrainerCallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Callback for debugging
class DebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(logs)

# Model and tokenizer initialization (adjust paths and settings as needed)
model_checkpoint = "outputs/microsoft/codebert-base-finetuned-codebertmlm/checkpoint-45"
model = RobertaForMaskedLM.from_pretrained(model_checkpoint, local_files_only=True)
# Move model to GPU if available
model = model.to(device)
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint, local_files_only=True)

def read_gzipped_jsonl(file_path):
    data = []
    with gzip.open(file_path, 'rt') as file:  # 'rt' mode for text reading
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data
import evaluate

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Dataset class from pytorch
class CodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        action_seq = item['action_seq']

        encoded_actions = self.tokenizer.encode_plus(
            action_seq,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')

        return {key: value.to(device) for key, value in encoded_actions.items()}

# Load the test data (adjust the path and method as per your original script)
gzipped_jsonl_file = 'dataset/evaluation_data.jsonl.gz'
# Define the read_gzipped_jsonl function or similar function to load your data
# ...
test_data = read_gzipped_jsonl(gzipped_jsonl_file)

test_data = test_data[:10000]

# Creating the test dataset
test_dataset = CodeDataset(test_data, tokenizer)

# Creating Datacollator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

num_gpus = torch.cuda.device_count()  # Automatically detects the number of GPUs available
print(num_gpus)

# Assuming per_device_train_batch_size is the batch size per GPU
per_device_train_batch_size = 32  # Adjust based on your GPU memory

# Effective total batch size across all GPUs
total_train_batch_size = per_device_train_batch_size * num_gpus

# Calculate steps_per_epoch as the number of samples divided by the total batch size

# Update training arguments
training_args = TrainingArguments(
    output_dir=f"./outputs/{model_checkpoint}-finetuned-codebertmlm-epoch",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    learning_rate=5e-5,
    weight_decay=0.01,
    per_device_eval_batch_size=8,  # Adjust based on your GPU memory
    num_train_epochs=1,
    push_to_hub=False,
    report_to='none',
eval_accumulation_steps=5,
    # Additional arguments for multi-GPU setup
    # ...
)



# Callback for debugging
class DebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(logs)


trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[DebugCallback()]
)

trainer.train()
