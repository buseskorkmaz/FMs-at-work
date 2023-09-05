from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
import argparse

def encode_text(row):
    # Preprocess the text
    text = row['description']
    print(text)
    # text = text.replace('\n', ' ').replace(',', ' ')

    # Load pre-trained model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize and pad the text to a maximum length of 512 tokens
    input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=256, truncation=True, padding='max_length')

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


def main():
    parser = argparse.ArgumentParser(description="Process the index.")
    parser.add_argument("--index", type=int, required=True, help="The index to process")

    args = parser.parse_args()
    index = args.index

    # load dataset
    hiring_dataset = load_dataset("json", data_files='/dccstor/autofair/bias_llm/Bias-ILQL/data/workable/job_descriptions.json')['train']

    # dropna
    # hiring_dataset = hiring_dataset.filter(not_none)

    # split batches
    num_batches = 24
    batched_datasets = []
    length_of_dataset = len(hiring_dataset)

    for i in range(num_batches):
        start_index = int((i * length_of_dataset) / num_batches)
        end_index = int(((i + 1) * length_of_dataset) / num_batches)
        batched_datasets.append(hiring_dataset.shard(num_batches, i))

    # take the specified shard
    hiring_dataset = batched_datasets[index]

    # Apply the function to the dataset
    hiring_dataset = hiring_dataset.map(encode_text)
    hiring_dataset.save_to_disk(f"processed_hiring_embeddings_{index}")

main()