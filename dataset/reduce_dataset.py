import json

jsonl_file = 'output_data_filtering<200.jsonl'

# New file to store the reduced dataset
filtered_jsonl_file = 'output_data.jsonl'

# Processing the file
with open(jsonl_file, 'r') as infile, open(filtered_jsonl_file, 'w') as outfile:
    for i, line in enumerate(infile):
        if i % 3 == 0:
            json_obj = json.loads(line)
            # Extract only the action_seq key
            action_seq = json_obj.get("action_seq", [])
            # Write the action_seq data to the new file
            json.dump({"action_seq": action_seq}, outfile)
            outfile.write('\n')
        else:
            continue


print()
print("Data processing complete. Reduced file created:", filtered_jsonl_file)