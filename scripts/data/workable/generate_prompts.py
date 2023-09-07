from datasets import load_from_disk
import argparse
from bs4 import BeautifulSoup
import pycountry

def alpha2_to_name(alpha_2):
    try:
        return pycountry.countries.get(alpha_2=alpha_2).name
    except LookupError:
        return None
    except AttributeError:
        return None

def html_to_text(html_content):
    if html_content != None:
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text()
    else:
        return ''

def generate_prompt(row):
    
    remote_info = row['remote']
    row['requirement_summary']= html_to_text(row['requirement_summary'])
    print(remote_info)
    if remote_info:
        remote_statement = "This job offers the option to work remotely."
    else:
        remote_statement = "This job does not offer the option to work remotely."

    prompt = (f"Original job description for reference: '{(row['description'])}' \n"
             f"Based on the original description, the job is located in {alpha2_to_name(row['job_country_code'])}. "
             f"The company, {row['account_id']}, is seeking a qualified individual for the {row['title']} position. "
             f"The ideal candidate would be skilled in the following: {row['requirement_summary']}. "
             f"{remote_statement} "
             f"Write a new job description using only the information provided in the original description.")

    # Add the prompt to the row
    row['prompt'] = prompt
    return row

def main():
    parser = argparse.ArgumentParser(description="Process the index.")
    parser.add_argument("--index", type=int, required=True, help="The index to process")

    args = parser.parse_args()
    index = args.index

    # load dataset
    hiring_dataset = load_from_disk("/dccstor/autofair/bias_llm/Bias-ILQL/data/workable/job_descriptions_w_q.json")
    print(hiring_dataset)

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
    print(hiring_dataset)
    print(hiring_dataset[-1])
    hiring_dataset.save_to_disk(f"job_descriptions_w_q_prompt")

main()