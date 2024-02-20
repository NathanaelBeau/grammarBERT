import gzip
import json

# Original reduced file
jsonl_file = 'output_data.jsonl'

# Compressed file name
compressed_file = 'output_data.jsonl.gz'

# Compressing the file
with open(jsonl_file, 'rb') as f_in:
    with gzip.open(compressed_file, 'wb') as f_out:
        f_out.writelines(f_in)

print("Compression complete. Compressed file created:", compressed_file)