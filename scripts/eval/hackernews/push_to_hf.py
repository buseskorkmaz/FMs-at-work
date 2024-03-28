from datasets import load_from_disk, concatenate_datasets
import logging
import os
from datasets import ClassLabel, Value
import json

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging("concatenate_and_push.log")
    
    logging.info("Collecting processed batch directories...")
    processed_batch_dirs = [d for d in os.listdir() if d.startswith("biasinbios_generated_")]
    logging.info(str(processed_batch_dirs))
    logging.info("Loading processed batches...")
    processed_batches = [load_from_disk(d) for d in processed_batch_dirs]
    for idx, batch in enumerate(processed_batches):
        # Convert nested dictionaries to JSON strings
        processed_batches[idx] = batch.map(lambda x: {
            'male_profession_percentages': json.dumps(x['male_profession_percentages']),
            'female_profession_percentages': json.dumps(x['female_profession_percentages']),
            'male_selected_profession_percentages': json.dumps(x['male_selected_profession_percentages']),
            'female_selected_profession_percentages': json.dumps(x['female_selected_profession_percentages']),
            **{k: v for k, v in x.items() if k not in ['male_profession_percentages', 'female_profession_percentages', 'male_selected_profession_percentages', 'female_selected_profession_percentages']}
        })

    logging.info("Concatenating processed batches...")
    concatenated_dataset = concatenate_datasets(processed_batches)
    logging.info(str(len(concatenated_dataset)))
    concatenated_dataset.push_to_hub("buseskorkmaz/impact_ratio_calc_filtered_generated_mrr", private=True)
    logging.info("Dataset pushed to Hugging Face Hub.")

if __name__ == "__main__":
    main()


