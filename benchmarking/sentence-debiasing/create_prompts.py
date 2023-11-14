import time
from datasets import load_dataset
from perspective_api import PerspectiveApiScorer
from googleapiclient.errors import HttpError
import json

def load():
    candidates = load_dataset("buseskorkmaz/wants_to_be_hired_gendered")["train"]
    text = load_dataset("buseskorkmaz/hiring_w_q_context_256_filtered", split="train")
    print(text['prompt'][0])
    return candidates, text

def read_idxs():
    with open('eval_idxs.json', 'r') as f:
        data = json.load(f)

    print(data)
    return data[:400]


def create_prompts_json():
    _, original_job_descriptions = load()
    eval_idxs = read_idxs()
    original_job_descriptions = original_job_descriptions.select(eval_idxs)
    scorer = PerspectiveApiScorer(api_key="AIzaSyAf-sAxVuYPVxAiQx0OmtXci3nsRSLuFJ4")

    with open('job_descriptions_prompts.jsonl', 'w') as outfile:
        for description in original_job_descriptions:
            while True:
                try:
                    scores = scorer.get_scores(input_text=description['text'])
                    # Append the job description to the scores
                    scores["text"] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. \n Instruction: {description['prompt']} \n Response: "
                    print(len(scores))
                    # Write the combined data to the jsonl file
                    outfile.write(json.dumps(scores) + '\n')
                    break
                except HttpError as e:
                    if e.resp.status == 429:
                        print("Rate limit exceeded. Sleeping for 60 seconds before retrying...")
                        time.sleep(60)  # Sleep for 60 seconds
                    else:
                        print(f"Unexpected error: {e}. Skipping this description.")
                        break

create_prompts_json()

