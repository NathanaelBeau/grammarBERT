import random

# Set the random seed to a specific number
random.seed(42)

from transformers import RobertaForMaskedLM, RobertaTokenizer, DataCollatorForLanguageModeling, TrainingArguments, \
    Trainer, TrainerCallback

from datasets import load_dataset
import torch

from asdl.ast_operation import Grammar, GrammarRule, ReduceAction
import evaluate

accuracy = evaluate.load("evaluate/metrics/accuracy/accuracy.py")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Flatten the tensors to 1D lists
    predictions = predictions.flatten().tolist()
    labels = labels.flatten().tolist()
    return accuracy.compute(predictions=predictions, references=labels)

def transform_to_id(examples):
    input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in examples['action_seq']]
    return {
        'input_ids': input_ids,
    }

def tokenize_function(examples):
    # Initialize lists for each column
    input_ids = []
    attention_masks = []
    special_tokens_masks = []

    # Iterate over each example
    for x in examples['input_ids']:
        # Decode and then tokenize with special token mask
        input_with_mask = tokenizer(tokenizer.decode(x), return_special_tokens_mask=True)

        # Append the results to the respective lists
        input_ids.append(input_with_mask['input_ids'])
        attention_masks.append(input_with_mask['attention_mask'])
        special_tokens_masks.append(input_with_mask['special_tokens_mask'])

    # Return a dictionary
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'special_tokens_mask': special_tokens_masks
    }

def padd_derivations(examples):
    pad_token_id = tokenizer.pad_token_id  # Make sure your tokenizer has a pad token

    # Pad each example individually
    padded_examples = {k: [] for k in examples.keys()}
    for i in range(len(examples['input_ids'])):
        for k in examples.keys():
            # Pad or truncate the example
            example = examples[k][i]
            padded_length = len(example)
            if padded_length > max_length:
                padded_example = example[:max_length]  # Truncate if longer than max_length
            else:
                padded_example = example + [pad_token_id] * (max_length - padded_length)  # Pad if shorter
            padded_examples[k].append(padded_example)

    # Set labels to be the same as input_ids
    padded_examples["labels"] = padded_examples["input_ids"].copy()
    # print(len(padded_examples["labels"][0]))

    return padded_examples


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


model_checkpoint = "microsoft/codebert-base"
model = RobertaForMaskedLM.from_pretrained(model_checkpoint, local_files_only=True)
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint, local_files_only=True)

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

dataset = load_dataset("json", data_files="dataset/output_data.json.gz")

dataset = dataset['train']

dataset = dataset.train_test_split(test_size=0.1)

dataset = dataset.map(transform_to_id, batched=True, num_proc=4, remove_columns=["action_seq"])

dataset = dataset.map(tokenize_function, batched=True, num_proc=4)

max_length = 256   # Set your desired max length

dataset = dataset.map(
    padd_derivations,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

dataset.save_to_disk("dataset/hf_dataset")
