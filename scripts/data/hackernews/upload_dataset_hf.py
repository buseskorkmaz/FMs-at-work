import glob
from datasets import load_from_disk, concatenate_datasets, load_dataset
import argparse
import typing as T

def upload_multiple_files(
    # directory_pattern:str = 'data/hackernews/hiring_technologies/processed_hiring_*',
    # dataset_name:str = "buseskorkmaz/hackernews_hiring_technologies_combined",
    # private:bool = True,
    directory_pattern:str = 'processed_profile_embeddings*',
    dataset_name:str = "buseskorkmaz/wants_to_be_hired_gendered",
    private:bool = True

):

    # The pattern to match the directories
    # Use glob to find all directories that match the pattern
    directories = glob.glob(directory_pattern)

    # Load each directory into a huggingface dataset
    datasets = [load_from_disk(directory) for directory in directories]
    print("LIST OF DATASETS:\n", datasets)

    # Concatenate all the datasets together
    combined_dataset = concatenate_datasets(datasets)
    print("COMBINED:\n" , combined_dataset)
    print(combined_dataset[512])

    # Now combined_dataset contains the combined rows from all directories
    combined_dataset.push_to_hub(dataset_name, private=private)

    return

def upload_single_file(
    directory:str = 'processed_hiring_prompts_w_context',
    dataset_name:str = "buseskorkmaz/hackernews_hiring_prompts_w_context",
    private:bool = True
):
    # Load the dataset from disk
    dataset = load_from_disk(directory)
    print(dataset)

    # Push the dataset to hub
    dataset.push_to_hub(dataset_name, private=private)

    return

def not_none(example):
    return example['text'] is not None


def concatenate_columns(
        dataset_names:T.List = [],
        final_name: str = "hiring", 
):  
    
    # print("weird")
    # Load the two datasets from the HuggingFace hub
    # hiring_location = load_dataset('buseskorkmaz/hackernews_hiring_location_combined', split="train")
    # hiring_technologies = load_dataset('buseskorkmaz/hackernews_hiring_technologies_combined', split="train")
    # print(hiring_location)
    # print(hiring_technologies)

    # The pattern to match the directories
    # Use glob to find all directories that match the pattern
    # directory_pattern = 'data/hackernews/hiring_q_values/processed_q_values_*'
    # directories = glob.glob(directory_pattern)
    # print(directories)

    # # Load each directory into a huggingface dataset
    # datasets = [load_from_disk(directory) for directory in directories]
    # print("LIST OF DATASETS:\n", datasets)

    # # Concatenate all the datasets together
    # combined_dataset = concatenate_datasets(datasets)
    # print("COMBINED:\n" , combined_dataset)
    # print(combined_dataset[0])

    rl_dataset = load_dataset("buseskorkmaz/hackernews_new_q_values_10")["train"]
    # load dataset
    prompt_dataset = load_dataset("buseskorkmaz/hackernews_hiring_prompts")["train"]

    # dropna
    # user_profile_dataset = user_profile_dataset.filter(not_none)

    # 
    # # title = load_dataset("buseskorkmaz/hackernews_hiring_title_combined")["train"]
    # print(user_profile_dataset)
    # print(combined_dataset)

    # # Ensure the datasets are of the same size
    # assert len(rl_dataset) == len(prompt_dataset), "Datasets are of different sizes!"

    # # Add the 'technologies' feature from the second dataset to the first
    # # hiring_location = hiring_location.map(lambda example, idx: {'technologies': hiring_technologies[idx]['technologies']}, with_indices=True)
  
    # # Create a dictionary from the second dataset for efficient lookup
    tech_dict = {item['text']: item['prompt'] for item in prompt_dataset}

    # Add the 'technologies' feature from the second dataset to the first
    # def add_technologies(example):
    #     # Get the technologies corresponding to the text, or None if not found
    #     example['technologies'] = tech_dict.get(example['text'], None)
    #     return example

    def add_new_column(example, new_column_name):
        # Get the technologies corresponding to the text, or None if not found
        example[new_column_name] = tech_dict.get(example['text'], None)
        return example

    # Now hiring_location contains the combined rows from both datasets
    rl_dataset = rl_dataset.map(lambda x: add_new_column(x, new_column_name="prompt"))
    print(rl_dataset)
    print(rl_dataset[32])
    
    # Push to hub
    rl_dataset.push_to_hub("hackernews_new_q_values_10_no_context", private=True)
    
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Upload single or multiple files.")
    parser.add_argument("--upload_option", type=str, help="Set multiple if multiple files will be uploaded otherwise set single")
    parser.add_argument("--combine_datasets", type=str, help="Set 'yes' if the datasets are needed to concatenated")
    args = parser.parse_args()
    if args.upload_option:
        upload_option = args.upload_option
    else:
        upload_option = None
    print(upload_option)

    if upload_option != None:
        if upload_option.lower() == "multiple":
            upload_multiple_files()
        else:
            upload_single_file()
    
    if args.combine_datasets == "yes":
        concatenate_columns()

