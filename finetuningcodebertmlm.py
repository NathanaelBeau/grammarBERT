from transformers import RobertaForMaskedLM, RobertaTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer, TrainerCallback
import json
from torch.utils.data import Dataset, DataLoader
import random
import torch
import wandb
from asdl.ast_operation import Grammar, GrammarRule, ReduceAction



# Model and tokenizer initialization
model_checkpoint = "microsoft/codebert-base-mlm"
model = RobertaForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)


# loading data from jsonl file
jsonl_file = 'dataset/output_data.jsonl'
with open(jsonl_file, 'r') as file:
    full_data = [json.loads(line) for line in file]
# with open(jsonl_file) as f:
#     full_data = json.load(f)


#Checking loaded data
print(f"Number of data examples loaded : {len(full_data)}")


# retrieve all grammar actions in a list
asdl_text = open('./asdl/PythonASDLgrammar3,9.txt').read()
grammar, _, _ = Grammar.from_text(asdl_text)
act_list = [GrammarRule(rule.constructor.name, rule.type.name, rule.fields) for rule in grammar]
assert (len(grammar) == len(act_list))
Reduce = ReduceAction('Reduce')
act_dict = dict([(act.label, act) for act in act_list])

# # increase the vocabulary of Bert model and tokenizer
new_tokens = act_dict
num_added_toks = tokenizer.add_tokens(new_tokens)

model.resize_token_embeddings(len(tokenizer))


# Dataset class from pytorch
class CodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code, action_seq = self.data[idx]
        encoded_code = self.tokenizer.encode_plus(
            code,
            add_special_tokens=True, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt')
        
        encoded_actions = self.tokenizer.encode_plus(
            action_seq, 
            add_special_tokens=True, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt')
        

        return {'input_ids': encoded_code['input_ids'].squeeze(), 
                'attention_mask': encoded_code['attention_mask'].squeeze(), 
                'labels': encoded_actions['input_ids'].squeeze()
            }




# train test split
random.shuffle(full_data)
train_size = int(0.8 * len(full_data))
train_data, test_data = full_data[:train_size], full_data[train_size:]


#print training and testing dataset
print(f"Training dataset size : {len(train_data)}")
print(f"Testing dataset size: {len(test_data)}")



# Creating DataLoaders
train_dataset = CodeDataset(train_data, tokenizer)
test_dataset = CodeDataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


# Creating Datacollator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


################print mask examples#################
# samples = [dataset[i] for i in range(2)]
# for sample in samples:
#     input_ids = sample['input_ids']
#     print(f"\n'>>> {tokenizer.decode(input_ids)}'")

# for chunk in data_collator(samples)["input_ids"]:
#     print(f"\n'>>> {tokenizer.decode(chunk)}'")


###############idea: whole word masking= second data collector?##########################


# train_size = 50
# test_size = int(0.1 * train_size)

# downsampled_dataset = dataset["train"].train_test_split(
#     train_size=train_size, test_size=test_size, seed=42
# )
# downsampled_dataset

####################Dataset dict from huggingface not used because we use dataset class from pytorch#############
# DatasetDict({
#     train: Dataset({
#         features: ['code', 'action_seq'],
#         num_rows: 40
#     })
#     test: Dataset({
#         features: ['code', 'action_seq'],
#         num_rows: 10
#     })
#     unsupervised: Dataset({
#         features: ['code', 'action_seq'],
#         num_rows: 00
#     })
# })

# DatasetDict({
#     train: Dataset({
#         features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
#         num_rows: 10000
#     })
#     test: Dataset({
#         features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
#         num_rows: 1000
#     })
# })




###########Training arguments and trainer config################################


training_args = TrainingArguments(
    output_dir=f"{model_checkpoint}-finetuned-codebertmlm",
    evaluation_strategy="epoch", 
    learning_rate=2e-5, 
    weight_decay=0.01, 
    per_device_train_batch_size=2, 
    per_device_eval_batch_size=2, 
    num_train_epochs=3,
    push_to_hub=False,  
    fp16=False,
    logging_steps=len(train_loader)
    
)



# Callback for debugging
class DebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(logs)



trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_loader.dataset, 
    eval_dataset=test_loader.dataset, 
    data_collator=data_collator, 
    tokenizer=tokenizer, 
    callbacks=[DebugCallback()]

)




trainer.train()