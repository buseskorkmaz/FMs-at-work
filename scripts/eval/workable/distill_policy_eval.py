import argparse
import pickle as pkl
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../src'))
import dill
import re
from datasets import load_dataset, load_from_disk

def extract_text(input_string):
    pattern = r"(?<=parent:)(.*?)(?=comment:)"
    matches = re.findall(pattern, input_string, re.DOTALL)
    return [match.strip() for match in matches]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str)
    args = parser.parse_args()

    with open(args.eval_file, 'rb') as f:
        d = dill.load(f)

    # print(d)
    # print("=="*50)
    # print(d['eval_dump'])
    print(str(d['eval_dump']['results'][2][0]))
    prompts = []
    for item in d['eval_dump']['results']:
        if sum(map(lambda x: x[2], item[1])) != -200:
            prompts.append(extract_text(str(item[0]))[0])
    print(prompts[2])
    print(len(prompts))

    rs = [sum(map(lambda x: x[2], item[1])) for item in d['eval_dump']['results'] if sum(map(lambda x: x[2], item[1])) != -200.0]
    # ent = [-item for item in d['eval_dump']['entropies']]
    ent = []
    for idx, dump_item in enumerate( d['eval_dump']['results']):
        # print(idx)
        results_item = d['eval_dump']['results'][idx]
        ent_item = d['eval_dump']['entropies'][idx]
        if sum(map(lambda x: x[2], results_item[1])) != -200.0:
            ent.append(-ent_item)

    print(max(rs), min(rs))
    # print(rs)
    mean_r = np.mean(rs)
    std_r = np.std(rs)
    st_err_r = std_r / np.sqrt(len(rs))
    mean_ent = np.mean(ent)
    std_ent = np.std(ent)
    st_err_ent = std_ent / np.sqrt(len(ent))
    print(f'reward: {mean_r} +- {st_err_r}')
    print(f'entropy: {mean_ent} +- {st_err_ent}')
    print(len(ent), len(rs))

    hiring_dataset = load_from_disk("/dccstor/autofair/bias_llm/Bias-ILQL/data/workable_rl_dataset/job_descriptions_w_q_prompt_eng")
    # print(hiring_dataset)
    # print(hiring_dataset["prompt"][0])
    items = [row for row in hiring_dataset]

    prompt2idx = {items[idx]['prompt']: idx for idx in range(len(items))}

    original_q_values = []

    for prompt in prompts:
        idx = prompt2idx[prompt]
        q_value = hiring_dataset[idx]['q_value']
        original_q_values.append(q_value)
    
    # print(rs)
    # print(original_q_values)

    mean_r = np.mean(original_q_values)
    std_r = np.std(original_q_values)
    st_err_r = std_r / np.sqrt(len(original_q_values))
    print(f'original reward: {mean_r} +- {st_err_r}')
    # print(original_q_values)
    assert len(original_q_values) == len(rs)