from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
import argparse
import re 

def get_remote_info(row):

    text = str(row["text"])
    text = re.sub("\n", '', text)
    
    if "no remote" in text.lower():
        remote = "No"
    else:
        remote = "Unknown"
    
    location = str(row["location"])

    if "remote" in location.lower():
        remote = "Yes"

    return remote

def generate_prompt(row):

    remote_info = get_remote_info(row)
    if remote_info == "Yes":
        remote_statement = "This job offers the option to work remotely."
    elif remote_info == "No":
        remote_statement = "This job does not offer the option to work remotely."
    else:
        remote_statement = "The remote work options for this job are currently unknown."

    prompt = f"The job is located in {row['location']}. The company, {row['company']}, is seeking a qualified individual for the {row['title']} position. The ideal candidate would be skilled in the following technologies: {row['technologies']}. {remote_statement} Write a detailed job description based on this information."

    # Add the prompt to the row
    row['prompt'] = prompt
    return row

def not_none(example):
    return example['text'] is not None

def main():
    parser = argparse.ArgumentParser(description="Process the index.")
    parser.add_argument("--index", type=int, required=True, help="The index to process")

    args = parser.parse_args()
    index = args.index

    # load dataset
    hiring_dataset = load_dataset("buseskorkmaz/backup_wo_emb", split="train")
    print(hiring_dataset)

    # dropna
    hiring_dataset = hiring_dataset.filter(not_none)

    # split batches
    num_batches = 1
    batched_datasets = []
    length_of_dataset = len(hiring_dataset)

    for i in range(num_batches):
        start_index = int((i * length_of_dataset) / num_batches)
        end_index = int(((i + 1) * length_of_dataset) / num_batches)
        batched_datasets.append(hiring_dataset.shard(num_batches, i))

    # take the specified shard
    hiring_dataset = batched_datasets[index]

    # Apply the function to the dataset
    hiring_dataset = hiring_dataset.map(generate_prompt)
    hiring_dataset.save_to_disk(f"processed_hiring_prompts_{index}")

main()