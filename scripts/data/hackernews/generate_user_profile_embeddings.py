from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
import argparse

def encode_text(row):
    gender= row['Gender']

    # Preprocess the text
    text = row['text']
    # text = text.replace('Location:', '').replace('Remote:', '').replace('Willing to relocate:', '').replace('Technologies:', '').replace('Resume:', '').replace('email:', '')
    text = text.replace('\n', ' ').replace(',', ' ')
    text = f"I identify my gender is {gender.lower()}. " + text
    print(text)

    # Load pre-trained model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize and pad the text to a maximum length of 512 tokens
    input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True, padding='max_length')

    # Convert to tensor
    input_ids = torch.tensor([input_ids])

    # Get the embeddings
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples

    # Get the embeddings of the '[CLS]' token, which represents the entire sentence
    sentence_embedding = last_hidden_states[0][0]

    # Convert the tensor to a list
    sentence_embedding = sentence_embedding.tolist()

    # Add the embedding to the row
    row['embedding'] = sentence_embedding
    return row

def not_none(example):
    return example['text'] is not None

def main():
    parser = argparse.ArgumentParser(description="Process the index.")
    parser.add_argument("--index", type=int, required=True, help="The index to process")
    print("Starting to embed user profiles ...")

    args = parser.parse_args()
    index = args.index

    # load dataset
    user_profile_dataset = load_dataset("buseskorkmaz/wants_to_be_hired")["wants_to_be_hired"]
    print(user_profile_dataset)

    # # dropna
    # user_profile_dataset = user_profile_dataset.filter(not_none)

    # split batches
    num_batches = 10
    batched_datasets = []
    length_of_dataset = len(user_profile_dataset)

    for i in range(num_batches):
        start_index = int((i * length_of_dataset) / num_batches)
        end_index = int(((i + 1) * length_of_dataset) / num_batches)
        batched_datasets.append(user_profile_dataset.shard(num_batches, i))

    # take the specified shard
    user_profile_dataset = batched_datasets[index]

    # Apply the function to the dataset
    user_profile_dataset = user_profile_dataset.map(encode_text)
    user_profile_dataset.save_to_disk(f"processed_profile_embeddings_{index}")

main()