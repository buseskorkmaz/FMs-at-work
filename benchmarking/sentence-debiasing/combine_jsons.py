import json
import os

# List of JSON file names
json_files = [
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_0.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_1.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_2.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_3.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_4.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_5.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_6.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_7.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_8.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_9.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_10.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_11.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_12.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_13.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_14.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_15.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_16.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_17.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_18.jsonl",
    "$HOME/FMs-at-work/benchmarking/sentence-debiasing/generated_debias_texts_gemma_19.jsonl"
]

# Combined JSON data
combined_data = []

# Iterate over each JSON file
for file_name in json_files:
    # Check if the file exists
    if os.path.exists(file_name):
        # Open the file and read the JSON data
        with open(file_name, 'r') as file:
            data = json.load(file)
            combined_data.extend(data)
    else:
        print(f"File not found: {file_name}")

# Write the combined data to a new JSON file
output_file = "home/bsk18/FMs-at-work/benchmarking/sentence-debiasing/combined_generated_debias_texts_gemma.json"
with open(output_file, 'w') as file:
    json.dump(combined_data, file)

print(f"Combined JSON data saved to {output_file}")