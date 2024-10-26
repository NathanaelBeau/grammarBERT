import torch
from transformers import RobertaTokenizer
from asdl.ast_operation import Grammar, GrammarRule, ReduceAction
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

def get_act_dict(path='./asdl/ASDL3.8.txt'):
    asdl_text = open(path).read()
    grammar, _, terminal_types = Grammar.from_text(asdl_text)
    act_list = [GrammarRule(rule.constructor.name, rule.type.name, rule.fields) for rule in grammar]
    Reduce = ReduceAction('Reduce')
    ReducePrimitif = ReduceAction('Reduce_primitif')
    act_dict = dict([(act.label, act) for act in act_list])
    act_dict[Reduce.label] = Reduce
    act_dict[ReducePrimitif.label] = ReducePrimitif
    return act_dict, terminal_types

def transform_to_id(examples, special_tokens=False):
    input_ids, attention_masks, special_tokens_masks = [], [], []
    for x in examples['action_seq']:
        if len(x) > 512:
            x = x[:510]
        input_ids_mask = tokenizer.convert_tokens_to_ids(['<s>'] + x + ['</s>'])
        special_tokens_mask = tokenizer.get_special_tokens_mask(input_ids_mask, already_has_special_tokens=True) if special_tokens else tokenizer.get_special_tokens_mask(input_ids_mask)
        input_ids.append(input_ids_mask)
        attention_masks.append([1] * len(input_ids_mask))
        special_tokens_masks.append(special_tokens_mask)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'special_tokens_mask': special_tokens_masks
    }

def preprocess_logits_for_metrics(logits, labels):
    return torch.argmax(logits, dim=-1)