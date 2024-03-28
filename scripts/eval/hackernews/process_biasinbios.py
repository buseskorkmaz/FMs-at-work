from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
import logging
import sys

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def encode_text(example):
    # Preprocess the text
    text = example['text']
    logging.info(f"Encoding text: {text[:50]}...")
    
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
    
    # Add the embedding to the example
    example['embedding'] = sentence_embedding
    logging.info(f"Encoded....")
    return example

def map_occupation(example):
    occupation_map = {
        0: 'accountant',
        1: 'architect',
        2: 'attorney',
        3: 'chiropractor',
        4: 'comedian',
        5: 'composer',
        6: 'dentist',
        7: 'dietitian',
        8: 'dj',
        9: 'filmmaker',
        10: 'interior_designer',
        11: 'journalist',
        12: 'model',
        13: 'nurse',
        14: 'painter',
        15: 'paralegal',
        16: 'pastor',
        17: 'personal_trainer',
        18: 'photographer',
        19: 'physician',
        20: 'poet',
        21: 'professor',
        22: 'psychologist',
        23: 'rapper',
        24: 'software_engineer',
        25: 'surgeon',
        26: 'teacher',
        27: 'yoga_teacher'
    }
    numerical_label = example['profession']
    example['profession_name'] = occupation_map.get(numerical_label, 'Unknown')
    return example

def process_batch(batch_index, dataset):
    batch = dataset.select(range(batch_index * batch_size, (batch_index + 1) * batch_size))
    logging.info(f"Processing batch {batch_index + 1}/{num_batches}")
    processed_batch = batch.map(encode_text, batched=False)
    processed_batch = processed_batch.map(map_occupation, batched=False)
    logging.info(f"Finished processing batch {batch_index + 1}/{num_batches}")
    return processed_batch

def main(batch_index):
    setup_logging(f"processing_batch_{batch_index}.log")
    
    logging.info("Loading dataset...")
    dataset = load_dataset("LabHC/bias_in_bios")["test"]
    
    logging.info("Renaming 'hard_text' to 'text'...")
    dataset = dataset.rename_column("hard_text", "text")
    dataset = dataset.rename_column("gender", "Gender")
    
    logging.info("Mapping gender values...")
    gender_map = {0: "Male", 1: "Female"}
    dataset = dataset.map(lambda example: {"Gender": gender_map[example["Gender"]]})
    
    logging.info("Processing batch...")
    processed_batch = process_batch(batch_index, dataset)
    
    logging.info("Saving processed batch...")
    processed_batch.save_to_disk(f"processed_batch_{batch_index}")
    
    logging.info("Processing completed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <batch_index>")
        sys.exit(1)
    
    batch_index = int(sys.argv[1])
    num_batches = 200
    batch_size = len(load_dataset("LabHC/bias_in_bios")["test"]) // num_batches
    
    main(batch_index)