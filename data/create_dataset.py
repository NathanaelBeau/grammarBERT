import json
import ast
import random
from datasets import load_dataset
from transformers import RobertaForMaskedLM, RobertaTokenizer
from asdl.ast_operation import ast2seq, Grammar, GrammarRule, ReduceAction
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from utils import transform_to_id, get_act_dict

# Constants
BATCH_SIZE = 100
NUM_THREADS = 4
TIMEOUT_SECONDS = 10
MAX_TOKENS = 712  # Define max token length
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Tokenizer and Model Initialization
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
model_checkpoint = "microsoft/codebert-base"
model = RobertaForMaskedLM.from_pretrained(model_checkpoint, local_files_only=True)
model.resize_token_embeddings(len(tokenizer))

def split_data(file_path, train_ratio=0.8, seed=42):
    # Ensure reproducibility
    random.seed(seed)

    # Read the filtered dataset
    with open(file_path, 'r') as input_file:
        data = [json.loads(line) for line in input_file]

    # Shuffle the data
    random.shuffle(data)

    # Calculate split index
    split_idx = int(len(data) * train_ratio)

    # Split the data
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    # Write training data
    with open('dataset/train_data.jsonl', 'w') as train_file:
        for entry in train_data:
            json.dump(entry, train_file)
            train_file.write('\n')

    # Write evaluation data
    with open('dataset/evaluation_data.jsonl', 'w') as eval_file:
        for entry in eval_data:
            json.dump(entry, eval_file)
            eval_file.write('\n')

    print(f"Data split into {len(train_data)} training samples and {len(eval_data)} evaluation samples.")


def get_data(filter_length=400):
    ds = load_dataset("bigcode/the-stack", data_dir="data/python", split="train", streaming=True)
    filtered_ds = (sample for sample in ds if len(sample['content'].split()) < filter_length)
    return filtered_ds

def preprocess_example(sample, act_dict, primitives):
    try:
        code = sample['content']
        python_ast = ast.parse(code)
        action_seq_grammar, _, _, _ = ast2seq(python_ast, act_dict, primitives=primitives)
        actions_seq = []
        for action in action_seq_grammar:
            if isinstance(action[0], (GrammarRule, ReduceAction)):
                actions_seq.append(action[0].label)
            else:
                actions_seq.extend(tokenizer.tokenize(str(action[0])))
                actions_seq.append('<end_primitive>')
        if len(actions_seq) > MAX_TOKENS:
            return None
        return {'code': code, 'action_seq': actions_seq}
    except Exception:
        return None

def preprocess_batch(batch, act_dict, primitives):
    return [preprocess_example(sample, act_dict, primitives) for sample in batch if preprocess_example(sample, act_dict, primitives)]

def preprocess_examples(dataset, act_dict, primitives):
    output_data = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        batch = []
        for index, sample in enumerate(dataset):
            batch.append(sample)
            if len(batch) == BATCH_SIZE:
                futures.append(executor.submit(preprocess_batch, batch, act_dict, primitives))
                batch = []
        if batch:
            futures.append(executor.submit(preprocess_batch, batch, act_dict, primitives))
        for future in futures:
            try:
                done, _ = wait([future], timeout=TIMEOUT_SECONDS, return_when=FIRST_COMPLETED)
                for f in done:
                    output_data.extend(f.result())
            except Exception as e:
                print(f"Error occurred: {e}")
    with open('data/output_data_filtering.jsonl', 'w') as output_file:
        for example in output_data:
            if example:
                json.dump(example, output_file)
                output_file.write('\n')


if __name__ == "__main__":
    act_dict, primitives = get_act_dict(path='./asdl/ASDL3.8.txt')
    dataset = get_data(filter_length=1000)

    # Preprocess dataset and save filtered data
    preprocess_examples(dataset, act_dict, primitives)

    # Split the filtered data into training and evaluation sets
    split_data('data/output_data_filtering.jsonl', train_ratio=0.8)

    # Load and map datasets for Hugging Face
    dataset_train = load_dataset("json", data_files="dataset/train_data.jsonl", split='train')
    dataset_eval = load_dataset("json", data_files="dataset/evaluation_data.jsonl", split='train')

    # Optionally add special tokens
    only_primitives_mode = False
    if only_primitives_mode:
        tokenizer.add_special_tokens({'additional_special_tokens': list(act_dict)})

    # Apply tokenization
    dataset_train = dataset_train.map(transform_to_id, batched=True, num_proc=4, remove_columns=["action_seq"],
                                      fn_kwargs={'special_tokens': only_primitives_mode})
    dataset_eval = dataset_eval.map(transform_to_id, batched=True, num_proc=4, remove_columns=["action_seq"],
                                    fn_kwargs={'special_tokens': only_primitives_mode})

    # Save datasets to disk
    dataset_train.save_to_disk("dataset/hf_dataset_train")
    dataset_eval.save_to_disk("dataset/hf_dataset_eval")