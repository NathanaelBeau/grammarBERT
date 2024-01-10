import json
import ast
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from asdl.ast_operation import ast2seq
from asdl.grammar import Grammar, GrammarRule, ReduceAction
from transformers import RobertaForMaskedLM, RobertaTokenizer

# Constante pour la taille des lots
BATCH_SIZE = 100
# Nombre de threads à utiliser pour le ThreadPoolExecutor
NUM_THREADS = 4

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

def get_data():
    ds = load_dataset("bigcode/the-stack", data_dir="data/python", split="train", streaming=True)
    # Ajouter un filtre pour ne garder que les exemples avec moins de 200 tokens (à adapter selon le besoin)
    filtered_ds = (sample for sample in ds if len(sample['content'].split()) < 200)
    return filtered_ds

def preprocess_example(sample, act_dict, primitives):
    try:
        code = sample['content']
        example = {'code': code}

        python_ast = ast.parse(code)
        action_seq_grammar, _, _, _ = ast2seq(python_ast, act_dict, primitives=primitives)
        # actions_seq = [
        #     action[0].label if isinstance(action[0], GrammarRule) or isinstance(action[0], ReduceAction)
        #     else tokenizer.tokenize(str(action[0]))
        #     for action in action_seq_grammar]
        actions_seq = []
        for action in action_seq_grammar:  # We need to tokenize with the codeBERT tokenizer the primitives values also
            if isinstance(action[0], GrammarRule) or isinstance(action[0], ReduceAction):
                actions_seq.append(action[0].label)
            else:
                tokenized = tokenizer.tokenize(str(action[0]))
                actions_seq.extend(tokenized)  # This adds each element of the list individually
        if len(actions_seq) > 512:
            return None
        example['action_seq'] = actions_seq
        return example
    except Exception as e:
        return None

def preprocess_batch(batch, act_dict, primitives):
    processed_batch = []
    for sample in batch:
        processed_example = preprocess_example(sample, act_dict, primitives)
        if processed_example:
            processed_batch.append(processed_example)
    return processed_batch

def preprocess_examples(dataset, act_dict, primitives):
    len_code = []
    pass_examples = 0
    output_data = []

    # Prétraiter les données par lots en utilisant ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Création d'une liste pour les futurs résultats
        futures = []
        batch = []

        for index, sample in enumerate(dataset):
            batch.append(sample)
            if len(batch) == BATCH_SIZE:
                futures.append(executor.submit(preprocess_batch, batch, act_dict, primitives))
                batch = []
            if index == 10_000_000:
                break

        # Ajouter le dernier lot si nécessaire
        if batch:
            futures.append(executor.submit(preprocess_batch, batch, act_dict, primitives))

        # Récupérer les résultats à mesure qu'ils sont terminés
        for future in as_completed(futures):
            processed_batch = future.result()
            output_data.extend(processed_batch)
            len_code.extend([len(example['action_seq']) for example in processed_batch if example])


    # Écriture des données traitées en bloc dans le fichier
    with open('dataset/output_data.jsonl', 'w') as output_file:
        for example in output_data:
            if example:  # Assurez-vous que l'exemple n'est pas None
                json.dump(example, output_file)
                output_file.write('\n')

def get_act_dict(path='./asdl/PythonASDLgrammar3,9.txt'):
    asdl_text = open(path).read()
    grammar, _, terminal_types = Grammar.from_text(asdl_text)
    act_list = [GrammarRule(rule.constructor.name, rule.type.name, rule.fields) for rule in grammar]
    assert (len(grammar) == len(act_list))
    Reduce = ReduceAction('Reduce')
    act_dict = dict([(act.label, act) for act in act_list])
    return act_dict, terminal_types
