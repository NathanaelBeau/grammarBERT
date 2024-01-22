import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

from transformers import RobertaForMaskedLM, RobertaTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer, TrainerCallback
import json
from torch.utils.data import Dataset, DataLoader
import random
import torch
import gzip
import numpy as np
import evaluate
from asdl.ast_operation import Grammar, GrammarRule, ReduceAction
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score



# Model and tokenizer initialization
model_checkpoint = "microsoft/codebert-base"
model = RobertaForMaskedLM.from_pretrained(model_checkpoint, local_files_only=True)
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint, local_files_only=True)


# Specify the path to your GZIP-compressed JSONL file
gzipped_jsonl_file = 'dataset/output_data.jsonl.gz'

# Function to read GZIP-compressed JSONL file
def read_gzipped_jsonl(file_path):
    data = []
    with gzip.open(file_path, 'rt') as file:  # 'rt' mode for text reading
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data

# loading data from GZIP-compressed JSONL file
full_data = read_gzipped_jsonl(gzipped_jsonl_file)

#Checking loaded data
print(f"Number of data examples loaded : {len(full_data)}")


# retrieve all grammar actions in a list
asdl_text = open('./asdl/PythonASDLgrammar3,9.txt').read()
grammar, _, _ = Grammar.from_text(asdl_text)
act_list = [GrammarRule(rule.constructor.name, rule.type.name, rule.fields) for rule in grammar]
assert (len(grammar) == len(act_list))
Reduce = ReduceAction('Reduce')
ReducePrimitif = ReduceAction('Reduce_primitif')
act_dict = dict([(act.label, act) for act in act_list])
act_dict[Reduce.label] = Reduce
act_dict[ReducePrimitif.label] = ReducePrimitif

# # increase the vocabulary of Bert model and tokenizer
new_tokens = list(act_dict)
tokenizer.add_tokens(new_tokens)

model.resize_token_embeddings(len(tokenizer))


# Dataset class from pytorch
class CodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
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
                'labels': encoded_actions['labels'].squeeze()
            }


        # # Convert action sequence tokens to IDs
        # action_seq_ids = self.tokenizer.convert_tokens_to_ids(action_seq)
        # # Ensure action_seq_ids length does not exceed max_length
        # action_seq_ids = [0] + action_seq_ids[:self.max_length - 1] + [1]
        #
        # # Pad with zeros if shorter than max_length
        # padding_length = self.max_length - len(action_seq_ids)
        # action_seq_ids.extend([self.tokenizer.pad_token_id] * padding_length)
        #
        #
        # return {'input_ids': encoded_code['input_ids'].squeeze(),
        #         'attention_mask': encoded_code['attention_mask'].squeeze(),
        #         'labels': torch.tensor(action_seq_ids, dtype=torch.int32)
        #     }


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

# full_data = full_data[:5]

# train test split
random.shuffle(full_data)
train_size = int(0.9 * len(full_data))
train_data, test_data = full_data[:train_size], full_data[train_size:]


#print training and testing dataset
print(f"Training dataset size : {len(train_data)}")
print(f"Testing dataset size: {len(test_data)}")



# Creating DataLoaders
train_dataset = CodeDataset(train_data, tokenizer)
test_dataset = CodeDataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# Creating Datacollator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Assuming each epoch has 'steps_per_epoch' steps and you want to evaluate every 'n' epochs
steps_per_epoch = len(train_loader)  # Number of batches in the training loader
n = 1  # Evaluate every 2 epochs

training_args = TrainingArguments(
    output_dir=f"./outputs/{model_checkpoint}-finetuned-codebertmlm",
    evaluation_strategy="steps",
    learning_rate=5e-5,
    weight_decay=0.01,
    save_strategy="steps",
    eval_steps=steps_per_epoch * n,  # Evaluate every 'n' epochs
    logging_steps=steps_per_epoch * n,  # Log every 'n' epochs
    save_steps=steps_per_epoch * n,  # Save every 'n' epochs
    # per_device_train_batch_size=2,
    # per_device_eval_batch_size=2,
    num_train_epochs=5,
    push_to_hub=False,  
    fp16=False,
    report_to='none'
)


# Callback for debugging
class DebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(logs)


trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_loader.dataset, 
    eval_dataset=test_loader.dataset,
    data_collator=data_collator, 
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[DebugCallback()]
)



trainer.train()