import json
import ast

import numpy as np

from datasets import load_dataset
from asdl.ast_operation import ast2seq
from asdl.grammar import Grammar, GrammarRule, ReduceAction


def get_data():
    ds = load_dataset("bigcode/the-stack", data_dir="data/python", split="train", streaming=True)
    return ds

def preprocess_examples(dataset):
    act_dict, primitives = get_act_dict()
    len_code = []
    with open('dataset/output_data.jsonl', 'w') as output_file:
        for index, sample in enumerate(iter(dataset)):
            try:
                code = sample['content']
                example = {'code': code}

                python_ast = ast.parse(code)
                action_seq_grammar, _, _, _ = ast2seq(python_ast, act_dict, primitives=primitives)
                actions_seq = [
                    action[0].label if isinstance(action[0], GrammarRule) or isinstance(action[0], ReduceAction)
                    else (str(action[0]))
                    for action in action_seq_grammar]

                len_code.append(len(actions_seq))

                example['action_seq'] = actions_seq
                json.dump(example, output_file)  # write processed example to file
                output_file.write('\n')  # add newline

            except:
                continue

        print('number of example', index)
        print('mean', np.mean(len_code))
        print('var', np.var(len_code))

def get_act_dict(path='./asdl/PythonASDLgrammar3,9.txt'):
    asdl_text = open(path).read()
    grammar, _, terminal_types = Grammar.from_text(asdl_text)
    act_list = [GrammarRule(rule.constructor.name, rule.type.name, rule.fields) for rule in grammar]
    assert (len(grammar) == len(act_list))
    Reduce = ReduceAction('Reduce')
    act_dict = dict([(act.label, act) for act in act_list])
    return act_dict, terminal_types

if __name__ == '__main__':

    dataset = get_data()

    preprocess_examples(dataset)

