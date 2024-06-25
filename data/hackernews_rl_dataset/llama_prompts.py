from transformers import pipeline
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
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

def rewrite_prompt(text):
    # Create the rewritten prompt as a list of dictionaries
    rewritten_prompt = [
        {"role": "system", "content": "You are a helpful assistant that generates fair job descriptions for given specifications."},
        {"role": "user", "content": text}
    ]
    return rewritten_prompt

def process_batch(dataset):
    logging.info(f"Processing dataset")
    rewritten_prompts = []
    for item in dataset['prompt']:
        rewritten_prompt = rewrite_prompt(item)
        rewritten_prompts.append(rewritten_prompt)
    dataset = dataset.add_column("messages_llama", rewritten_prompts)
    logging.info(f"Finished processing dataset")
    return dataset

def main():
    setup_logging(f"processing_dataset.log")
    logging.info("Loading dataset...")
    dataset = load_dataset("buseskorkmaz/cleaned_hiring_dataset_qval_w_gendered_mpnet_fixed_prompt")["train"]
    logging.info("Processing batch...")
    rewritten_prompts = process_batch(dataset)
    logging.info("Pushing to hub...")
    rewritten_prompts.push_to_hub(f"buseskorkmaz/cleaned_hiring_dataset_qval_w_gendered_mpnet_fixed_llama3_prompt", private=True)
    logging.info("Processing completed.")

if __name__ == "__main__":
    main()