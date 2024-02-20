import random

# Set the random seed to a specific number
random.seed(42)

from transformers import RobertaForMaskedLM, RobertaTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer, TrainerCallback

import torch

from asdl.ast_operation import Grammar, GrammarRule, ReduceAction
import evaluate
from datasets import load_dataset, load_from_disk


accuracy = evaluate.load("evaluate/metrics/accuracy/accuracy.py")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Flatten the tensors to 1D lists
    predictions = predictions.flatten().tolist()
    labels = labels.flatten().tolist()
    return accuracy.compute(predictions=predictions, references=labels)

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids



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

model_checkpoint = "microsoft/codebert-base"
model = RobertaForMaskedLM.from_pretrained(model_checkpoint, local_files_only=True)
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint, local_files_only=True, additional_special_tokens=new_tokens)

tokenizer.add_tokens(new_tokens)

model.resize_token_embeddings(len(tokenizer))

# dataset_train = load_from_disk('dataset/hf_dataset_train')
dataset_train = load_from_disk('dataset/hf_dataset_train')
dataset_eval = load_from_disk('dataset/hf_dataset_eval')
# dataset_eval = dataset_eval.select(range(100))
# print(dataset_eval[1])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
# data_collator = CustomDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=f"./outputs/{model_checkpoint}-finetuned-codebertmlm-epoch-train",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    report_to='none',
    logging_steps=1,
    num_train_epochs=5,
    # fp16=True,  # Enable if GPUs support FP16
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=2,  # batch size for evaluation
    max_steps=10#int(5e7)
)

# Callback for debugging
class DebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(logs)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_eval,
    eval_dataset=dataset_eval,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[DebugCallback()],
    data_collator=data_collator
)

trainer.train()