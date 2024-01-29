import json
import gzip
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import EvalPrediction

from transformers import RobertaForMaskedLM, RobertaTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer, TrainerCallback

# Callback for debugging
class DebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(logs)

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
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {metric.compute(predictions=predictions, references=labels)}
# # Define the compute_metrics function (copy from your original script)
# def compute_metrics(eval_pred: EvalPrediction):
#     predictions, labels = eval_pred
#     # Flatten the outputs and labels
#     predictions = np.argmax(predictions, axis=-1).flatten()
#     labels = labels.flatten()
#
#     # Compute accuracy, excluding the ignored index (-100 used for non-masked tokens)
#     mask = labels != -100
#     predictions = predictions[mask]
#     labels = labels[mask]
#
#     return {"accuracy": accuracy_score(labels, predictions)}

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

# Creating Datacollator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)


# Creating the test dataset
test_dataset = CodeDataset(test_data, tokenizer)

# Manual evaluation
model.eval()
model.to('cuda')

# Define training arguments (adjust as needed)
training_args = TrainingArguments(
    output_dir=f"./outputs/{model_checkpoint}-finetuned-codebertmlm",
    per_device_eval_batch_size=8,    # Adjust based on your GPU memory
eval_accumulation_steps=20,
evaluation_strategy="steps",
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
    compute_metrics=compute_metrics,
data_collator = data_collator,
callbacks = [DebugCallback()]
)

# Perform the evaluation
evaluation_results = trainer.evaluate()

# Print the evaluation results
print("Evaluation Results:", evaluation_results)