import json
import random

jsonl_file = 'output_data_filtering<200.jsonl'

# File for the reduced dataset
filtered_jsonl_file = 'output_data.jsonl'
# File for the evaluation dataset
evaluation_jsonl_file = 'evaluation_data.jsonl'

used_indices = set()
evaluation_indices = set()

# First, create the reduced dataset and track used indices
with open(jsonl_file, 'r') as infile, open(filtered_jsonl_file, 'w') as outfile:
    for i, line in enumerate(infile):
        if i % 3 == 0:
            used_indices.add(i)
            json_obj = json.loads(line)
            action_seq = json_obj.get("action_seq", [])
            json.dump({"action_seq": action_seq}, outfile)
            outfile.write('\n')

# Generate unique indices for evaluation set
while len(evaluation_indices) < 100000:
    index = random.randint(0, len(used_indices) * 3)  # Adjust the range according to your dataset size
    if index not in used_indices:
        evaluation_indices.add(index)

# Extract the evaluation examples
with open(jsonl_file, 'r') as infile, open(evaluation_jsonl_file, 'w') as outfile:
    for i, line in enumerate(infile):
        if i in evaluation_indices:
            json_obj = json.loads(line)
            action_seq = json_obj.get("action_seq", [])
            json.dump({"action_seq": action_seq}, outfile)
            outfile.write('\n')

print("Data processing complete. Evaluation file created:", evaluation_jsonl_file)