from transformers import RobertaForMaskedLM, RobertaTokenizer, TrainingArguments, Trainer
import json
import gzip
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import EvalPrediction


# Model and tokenizer initialization (adjust paths and settings as needed)
model_checkpoint = "outputs/microsoft/codebert-base-finetuned-codebertmlm/checkpoint-45"
model = RobertaForMaskedLM.from_pretrained(model_checkpoint, local_files_only=True)
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint, local_files_only=True)

def read_gzipped_jsonl(file_path):
    data = []
    with gzip.open(file_path, 'rt') as file:  # 'rt' mode for text reading
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data

# Define the compute_metrics function (copy from your original script)
def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    # Flatten the outputs and labels
    predictions = np.argmax(predictions, axis=-1).flatten()
    labels = labels.flatten()

    # Compute accuracy, excluding the ignored index (-100 used for non-masked tokens)
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]

    return {"accuracy": accuracy_score(labels, predictions)}

# Define the dataset class (copy from your original script)
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

        return {'input_ids': encoded_actions['input_ids'].squeeze(),
                'attention_mask': encoded_actions['attention_mask'].squeeze(),
                'labels': encoded_actions['input_ids'].squeeze()
                }

# Load the test data (adjust the path and method as per your original script)
gzipped_jsonl_file = 'dataset/evaluation_data.jsonl.gz'
# Define the read_gzipped_jsonl function or similar function to load your data
# ...
test_data = read_gzipped_jsonl(gzipped_jsonl_file)

test_data = test_data[:10000]


# Creating the test dataset
test_dataset = CodeDataset(test_data, tokenizer)

# Manual evaluation
model.eval()

# Define training arguments (adjust as needed)
training_args = TrainingArguments(
    output_dir=f"./outputs/{model_checkpoint}-finetuned-codebertmlm",
    per_device_eval_batch_size=32,    # Adjust based on your GPU memory
eval_accumulation_steps=20,
    push_to_hub=False,
    report_to='none',
    # Additional arguments for multi-GPU setup
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Perform the evaluation
evaluation_results = trainer.evaluate()

# Print the evaluation results
print("Evaluation Results:", evaluation_results)