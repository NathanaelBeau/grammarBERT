import random

# Set the random seed for reproducibility
random.seed(42)

from transformers import (
    RobertaForMaskedLM,
    RobertaTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
import torch
from asdl.ast_operation import Grammar, GrammarRule, ReduceAction
import evaluate
from datasets import load_from_disk, DatasetDict

# Initialize accuracy metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten().tolist()
    labels = labels.flatten().tolist()
    return accuracy.compute(predictions=predictions, references=labels)

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

# Grammar setup
asdl_text = open('asdl/ASDL3.9.txt').read()
grammar, _, _ = Grammar.from_text(asdl_text)
act_list = [GrammarRule(rule.constructor.name, rule.type.name, rule.fields) for rule in grammar]
assert (len(grammar) == len(act_list))
Reduce = ReduceAction('Reduce')
ReducePrimitif = ReduceAction('Reduce_primitif')
act_dict = dict([(act.label, act) for act in act_list])
act_dict[Reduce.label] = Reduce
act_dict[ReducePrimitif.label] = ReducePrimitif

# Add new tokens
new_tokens = list(act_dict) + ['<end_primitive>']
model_checkpoint = "microsoft/codebert-base"
model = RobertaForMaskedLM.from_pretrained(model_checkpoint, local_files_only=True)
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint, local_files_only=True, additional_special_tokens=new_tokens)
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

# Load dataset and split into train and dev
dataset_train = load_from_disk('data/hf_dataset_train')
split_datasets = dataset_train.train_test_split(test_size=0.2, seed=42)
train_dataset = split_datasets["train"]
dev_dataset = split_datasets["test"]

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=f"./outputs/{model_checkpoint}-finetuned-codebertmlm-epoch-train",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    report_to='none',
    logging_steps=1,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    max_steps=10
)

# Debug callback
class DebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(logs)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[DebugCallback()],
    data_collator=data_collator
)

trainer.train()
