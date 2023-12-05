from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline, AutoTokenizer,  DataCollatorForLanguageModeling, TrainingArguments, Trainer, TrainerCallback
import json
from torch.utils.data import Dataset, DataLoader
import random
import torch
import wandb



model_checkpoint = "microsoft/codebert-base-mlm"
model = RobertaForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)



# loading data from jsonl file
jsonl_file = 'dataset/output_data.jsonl'
with open(jsonl_file, 'r') as file:
    full_data = [json.loads(line) for line in file]

# Dataset class from pytorch
class CodeDataset(Dataset):
    def __init__(self, jsonl_file, model_checkpoint, max_length=512):
        self.data = full_data
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
            return_tensors='pt'
        )


        encoded_actions = self.tokenizer.encode_plus(
            action_seq,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded_code['input_ids'].squeeze(),
            'attention_mask': encoded_code['attention_mask'].squeeze(),
            'labels': encoded_actions['input_ids'].squeeze()
        }





# train test split

random.shuffle(full_data)


train_size = int(0.8 * len(full_data))
test_size = len(full_data) - train_size


train_data = full_data[:train_size]
test_data = full_data[train_size:]

print(f"Training dataset size : {len(train_data)}")
print(f"Training dataset test: {len(test_data)}")



train_dataset = CodeDataset(train_data, tokenizer)
test_dataset = CodeDataset(test_data, tokenizer)

train_loader = DataLoader(CodeDataset(train_data, tokenizer), batch_size=4, shuffle=True)
test_loader = DataLoader(CodeDataset(test_data, tokenizer), batch_size=4, shuffle=False)


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



##############train test split with hf method###############



# downsampled_dataset = dataset["train"].train_test_split(
#     train_size=train_size, test_size=test_size, seed=42
# )



# batch_size = 16
# # Show the training loss with every epoch
# logging_steps = len(dataset["train"]) // batch_size
# model_name = model_checkpoint.split("/")[-1]

# train_loader = DataLoader(CodeDataset(train_data, tokenizer), batch_size=16, shuffle=True)
# test_loader = DataLoader(CodeDataset(test_data, tokenizer), batch_size=16, shuffle=True)


###########Training arguments and trainer configs################################


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!a tester!!!!!!!!!!!!!!!!!!!!!!!!!!
# def generate_and_save_model_outputs(model, dataloader, tokenizer, file_path):
#     model.eval()  # Mettre le modèle en mode évaluation
#     with open(file_path, "w") as file:
#         with torch.no_grad():
#             for batch in dataloader:
#                 inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
#                 outputs = model.generate(**inputs)
#                 for output in outputs:
#                     decoded_output = tokenizer.decode(output, skip_special_tokens=True)
#                     file.write(decoded_output + "\n")


# class GenerateOutputsCallback(TrainerCallback):
#     def on_evaluate(self, args, state, control, **kwargs):
#         generate_and_save_model_outputs(kwargs['model'], test_loader, tokenizer, "model_generated_outputs.txt")

training_args = TrainingArguments(
    output_dir=f"{model_checkpoint}-finetuned-codebertmlm",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    push_to_hub=False,
    fp16=False,
    logging_steps=len(train_loader),
)




trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_loader.dataset,
    eval_dataset=test_loader.dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
    # callbacks=[GenerateOutputsCallback()]

)


trainer.train()
