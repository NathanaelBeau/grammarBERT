import random

# Set the random seed to a specific number
random.seed(42)

from transformers import RobertaForMaskedLM, RobertaTokenizer

from datasets import load_dataset
import torch

from asdl.ast_operation import Grammar, GrammarRule, ReduceAction

def transform_to_id(examples, special_tokens=False):
    # Initialize lists for each column
    input_ids = []
    attention_masks = []
    special_tokens_masks = []

    for x in examples['action_seq']:

        input_ids_mask = tokenizer.convert_tokens_to_ids(['<s>'] + x + ['</s>'])

        if special_tokens:
            special_tokens_mask = tokenizer.get_special_tokens_mask(
                input_ids_mask , already_has_special_tokens=True
            )
        else:
            special_tokens_mask = tokenizer.get_special_tokens_mask(
                input_ids_mask
            )
        # Append the results to the respective lists
        input_ids.append(input_ids_mask)
        attention_masks.append([1 for i in range(len(input_ids_mask))])
        special_tokens_masks.append(special_tokens_mask)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'special_tokens_mask': special_tokens_masks
    }



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
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint, local_files_only=True, add_special_tokens=True, return_special_tokens_mask=True)


# Update the model to match the new tokenizer size
model.resize_token_embeddings(len(tokenizer))

dataset_train = load_dataset("json", data_files="dataset/output_data.jsonl.gz", split='train')
dataset_eval = load_dataset("json", data_files="dataset/evaluation_data.jsonl.gz", split='train')


only_primitives_mode = False

if only_primitives_mode:
    # Specify the new tokens as additional special tokens
    tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})

    dataset_train = dataset_train.map(
        transform_to_id,
        batched=True,
        num_proc=4,
        remove_columns=["action_seq"],
        fn_kwargs={'special_tokens': True}  # Passing special_tokens=True to transform_to_id
    )
    dataset_eval = dataset_eval.map(
        transform_to_id,
        batched=True,
        num_proc=4,
        remove_columns=["action_seq"],
        fn_kwargs={'special_tokens': True}  # Passing special_tokens=True to transform_to_id
    )
else:
    dataset_train = dataset_train.map(transform_to_id, batched=True, num_proc=4, remove_columns=["action_seq"])
    dataset_eval = dataset_eval.map(transform_to_id, batched=True, num_proc=4, remove_columns=["action_seq"])


dataset_train.save_to_disk("dataset/hf_dataset_train_nofilter")
dataset_eval.save_to_disk("dataset/hf_dataset_eval_nofilter")
