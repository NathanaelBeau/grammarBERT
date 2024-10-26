import random
import torch
from transformers import (RobertaForMaskedLM, RobertaTokenizer, DataCollatorForLanguageModeling,
                          TrainingArguments, Trainer, TrainerCallback)
import evaluate
from datasets import load_from_disk
from asdl.ast_operation import Grammar, GrammarRule, ReduceAction
from sklearn.metrics import accuracy_score
import numpy as np

# Set the random seed for reproducibility
random.seed(42)

# Load accuracy metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(dim=-1).flatten().tolist()
    labels = labels.flatten().tolist()
    return accuracy.compute(predictions=predictions, references=labels)

def preprocess_logits_for_metrics(logits, labels):
    return torch.argmax(logits, dim=-1)

# Load and process ASDL grammar
asdl_text = open('asdl/ASDL3.8.txt').read()
grammar, _, _ = Grammar.from_text(asdl_text)
act_list = [GrammarRule(rule.constructor.name, rule.type.name, rule.fields) for rule in grammar]
Reduce = ReduceAction('Reduce')
ReducePrimitif = ReduceAction('Reduce_primitif')
act_dict = dict([(act.label, act) for act in act_list])
act_dict[Reduce.label] = Reduce
act_dict[ReducePrimitif.label] = ReducePrimitif

# Define new tokens and add to tokenizer
new_tokens = list(act_dict) + ['<end_primitive>']
model_checkpoint = "microsoft/codebert-base"
model = RobertaForMaskedLM.from_pretrained('outputs/microsoft/codebert-base-finetuned-codebertmlm-epoch-train/checkpoint-x') # Put the checkpoint of your model here
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint, additional_special_tokens=new_tokens)
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

# Load evaluation dataset and data collator
dataset_eval = load_from_disk('data/hf_dataset_eval')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Evaluation-only arguments
training_args = TrainingArguments(
    output_dir=f"./outputs/{model_checkpoint}-evaluation",
    evaluation_strategy="no",
    per_device_eval_batch_size=2,
    report_to='none'
)

class DebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(logs)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=dataset_eval,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[DebugCallback()],
    data_collator=data_collator
)

# Prediction
predict_output = trainer.predict(dataset_eval)
print(predict_output.metrics)
