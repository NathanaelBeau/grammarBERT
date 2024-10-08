import json
import ast
from datasets import load_dataset
from asdl.ast_operation import ast2seq
from asdl.grammar import Grammar, GrammarRule, ReduceAction
from transformers import RobertaForMaskedLM, RobertaTokenizer

from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

# Constante pour la taille des lots
BATCH_SIZE = 100
# Nombre de threads à utiliser pour le ThreadPoolExecutor
NUM_THREADS = 4


# Définir un délai d'attente pour le traitement de chaque lot en secondes
TIMEOUT_SECONDS = 10


tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

def get_data():
    ds = load_dataset("bigcode/the-stack", data_dir="data/python", split="train", streaming=True)
    # Ajouter un filtre pour ne garder que les exemples avec moins de 200 tokens (à adapter selon le besoin)
    filtered_ds = (sample for sample in ds if len(sample['content'].split()) < 400)
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
                actions_seq.append('<end_primitive>')
        if len(actions_seq) > 712:
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
    pass_examples = 0
    output_data = []

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        batch = []

        for index, sample in enumerate(dataset):
            print(index)
            batch.append(sample)
            if len(batch) == BATCH_SIZE:
                # Soumettre le lot pour le traitement
                future = executor.submit(preprocess_batch, batch, act_dict, primitives)
                futures.append(future)
                batch = []
            if index == 10_500_000:
                break

        if batch:
            futures.append(executor.submit(preprocess_batch, batch, act_dict, primitives))

        for future in futures:
            try:
                # Utiliser wait pour appliquer un délai d'attente à chaque future
                done, _ = wait([future], timeout=TIMEOUT_SECONDS, return_when=FIRST_COMPLETED)
                for f in done:
                    processed_batch = f.result()
                    output_data.extend(processed_batch)
            except Exception as e:
                print(f"Timeout or error occurred: {e}")

    with open('dataset/output_data_filtering<400.jsonl', 'w') as output_file:
        for example in output_data:
            if example:
                json.dump(example, output_file)
                output_file.write('\n')

def get_act_dict(path='./asdl/grammar_3.8.txt'):
    asdl_text = open(path).read()
    grammar, _, terminal_types = Grammar.from_text(asdl_text)
    act_list = [GrammarRule(rule.constructor.name, rule.type.name, rule.fields) for rule in grammar]
    assert (len(grammar) == len(act_list))
    Reduce = ReduceAction('Reduce')
    ReducePrimitif = ReduceAction('Reduce_primitif')
    act_dict = dict([(act.label, act) for act in act_list])
    act_dict[Reduce.label] = Reduce
    act_dict[ReducePrimitif.label] = ReducePrimitif
    return act_dict, terminal_types
