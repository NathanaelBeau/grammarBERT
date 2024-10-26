# grammarBERT

`grammarBERT` fine-tunes the `codeBERT` model on Python derivation sequences using a Masked Language Modeling (MLM) task. This approach allows `codeBERT` to handle both natural language and syntax-specific code tokens, enhancing its effectiveness in tasks like code generation and grammar-based programming tasks.

## Repository Structure
```bash 
├── asdl                  # Grammar definitions, AST operations, and transition system
│   ├── ASDL3.8.txt
│   ├── ASDL3.9.txt
│   ├── ast_operation.py
│   ├── grammar.py
│   └── utils.py
├── dataset               # Datasets and preprocessing scripts
│   ├── create_dataset.py
│   └── utils.py
├── README.md             
├── test.py               # Script for evaluating your own grammarBERT
└── train.py              # Script for training your own grammarBERT
```
## Training and Fine-Tuning

To train or fine-tune your own `grammarBERT` model, follow these steps:

1. **Prepare the Dataset**: Use the `data/create_dataset.py` script to parse Python code and generate AST derivation sequences. This will create the necessary data files for training.

```bash
python data/create_dataset.py
```
2. Train the Model: Run the train.py script to fine-tune the model on your dataset. You can customize training parameters such as learning rate, batch size, and number of epochs in the script.
```bash
python train.py
```
3.  Evaluate the Model: After training, use the test.py script to evaluate the model on your test dataset. This script provides accuracy metrics and optional debugging outputs.
```bash 
python test.py
```


## Methodology

We utilize the transition system outlined in this [ACL article](https://aclanthology.org/2022.findings-acl.173.pdf).


During preprocessing, the code is converted into a derivation sequence. In this process, grammar rules are mapped to a grammar-specific vocabulary, while terminal symbols are tokenized using the `codeBERT` tokenizer. Terminal tokens, representing leaf nodes, may consist of multiple subtokens (e.g., "enumerate" tokenizes as `['e', 'num', 'rate']`). To handle this, we introduce an additional symbol into the `codeBERT` vocabulary, indicating the completion of terminal symbol prediction. This symbol plays a distinct role from the REDUCE action, which signifies the end of a grammar rule. While REDUCE marks the conclusion of a rule, the new token ensures the proper termination of terminal sequences before the model predicts subsequent terminals.  We have integrated grammar rules into the `codeBERT` pretrained vocabulary to improve its representation of both grammar rules and terminal symbols.

## Customization

- You can modify `create_dataset.py` to use your own dataset and your Python version.
- To use a different Python version, simply copy the relevant grammar rules into the `asdl/` directory from [Python's AST documentation](https://docs.python.org/3/library/ast.html).
- The training configuration can be adjusted directly in `train.py`.

## Test it

You can try our version of `grammarBERT` for Python 3.8, trained on the Stack Dataset, available on the Hugging Face hub:

```python
from transformers import RobertaForMaskedLM, RobertaTokenizer
from asdl.ast_operation import ast2seq

# Load the pre-trained grammarBERT model and tokenizer
model = RobertaForMaskedLM.from_pretrained("Nbeau/grammarBERT")
tokenizer = RobertaTokenizer.from_pretrained("Nbeau/grammarBERT")

code_snippet = "def enumerate_items(items):"
derivation_sequence = ast2seq(code_snippet)
input_ids = tokenizer.encode(code_snippet, return_tensors='pt')

# Predict masked tokens or fine-tune the model as needed
outputs = model(input_ids)
```