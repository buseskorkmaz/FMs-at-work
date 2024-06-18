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
    # processed_batch_dirs = [d for d in os.listdir() if d.startswith("occupations_processed_batch")]
    processed_batch_dirs = sorted([d for d in os.listdir() if d.startswith("wants_to_hired_gendered_sentence_embeddings_gendered_distilroberta")], key=lambda x: int(x.split("_")[-1])) 
                                #    and "_test" in d], 
                                  
    logging.info(str(processed_batch_dirs))
    logging.info("Loading processed batches...")
    processed_batches = [load_from_disk(d) for d in processed_batch_dirs]
    # for idx, batch in enumerate(processed_batches):
    #     # Convert nested dictionaries to JSON strings
    #     processed_batches[idx] = batch.map(lambda x:{
    #                                     #    {"sr_remote": float(x['sr_remote']),
    #         'female_profession_percentages': json.dumps(x['female_profession_percentages']),
    #         'male_selected_profession_percentages': json.dumps(x['male_selected_profession_percentages']),
    #         'female_selected_profession_percentages': json.dumps(x['female_selected_profession_percentages']),
    #         **{k: float(v) if isinstance(v, (int, float)) else v for k, v in x.items() 
    #            if k not in ['male_profession_percentages', 'female_profession_percentages', 'male_selected_profession_percentages', 'female_selected_profession_percentages']}
    #     })

    logging.info("Concatenating processed batches...")
    concatenated_dataset = concatenate_datasets(processed_batches)
    logging.info(str(len(concatenated_dataset)))
    concatenated_dataset.push_to_hub("buseskorkmaz/cleaned_hiring_dataset_qval_w_gendered_distilroberta", private=True)
    logging.info("Dataset pushed to Hugging Face Hub.")

if __name__ == "__main__":
    main()

# from huggingface_hub import HfApi
# from transformers import shard_checkpoint

# shard_checkpoint(
#     checkpoint_path="path/to/your/80gb/checkpoint", 
#     max_shard_size="10GB",
#     output_dir="path/to/output/sharded/checkpoint"
# )

# api = HfApi()
# api.upload_file(
#     path_or_fileobj="/gpfs/home/bsk18/FMs-at-work/outputs/hackernews/llama/frozen_512_bios_no_offload/model_45055.pkl",
#     path_in_repo="model_45055.pkl",
#     repo_id="buseskorkmaz/llama-iql-model-bios",
#     repo_type="model",
# )
# api.upload_file(
#     path_or_fileobj="/gpfs/home/bsk18/FMs-at-work/outputs/hackernews/llama/frozen_512_hn_no_offload/config.json",
#     path_in_repo="config.json",
#     repo_id="buseskorkmaz/llama-iql-model-bios",
#     repo_type="model",
# )

