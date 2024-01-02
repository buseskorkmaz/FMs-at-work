import os
import csv
from typing import Callable, List, Optional
from utils.cache import Cache
from datasets import load_dataset
import re

class HackernewsData:
    def __init__(self,
                 indexes: Optional[List[int]], 
                 reward_f: Optional[Callable[[str], float]], 
                 reward_cache: Optional[Cache]=None, 
                 reward_shift: float=0.0, 
                # not sure about them
                 reward_scale: float=1.0):
        
        rl_dataset = load_dataset("buseskorkmaz/hiring_w_q_context_256_filtered", split="train")
        print(rl_dataset)
        # print(len(rl_dataset))
        # print(rl_dataset[0])
        items = [row for row in rl_dataset]
        # print("Indexes:", indexes)
        if indexes is not None:
            items = [items[idx] for idx in indexes]
        self.info = ("huggingface", len(indexes))
        self.reward_cache = reward_cache
        if self.reward_cache is None:
            self.reward_cache = Cache()

        # remove links <a> and </a> are special tokens
        def remove_links(text):
            clean_text = re.sub('<a.*?</a>', '', text)
            clean_text = clean_text.replace('<a href="', '')
            clean_text = clean_text.replace('www.', '')
            clean_text = clean_text.replace("</a>", '')
            clean_text = clean_text.replace('"', '')
            return clean_text

        self.data = [remove_links(item['text']) for item in items]
        self.parent_data = [remove_links(item['prompt']) for item in items]
        self.gt_scores = [-100 if float(item['q_value']) == -1000.0 else float(item['q_value']) for item in items]
        # self.location = [item['location'] for item in items]
        # self.embedding = [item['embedding'] for item in items]
        # self.remote = [item['remote'] for item in items]
        # self.relocate = [item['relocate'] for item in items]
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale
        print("INITIALIZED WITH REWARD F", reward_f)
        self.reward_f = reward_f

        print(self.data[12])
        print(self.parent_data[12])
        print(self.gt_scores[12])
        print("Uploaded dataset length:", len(self.data))

    def __getitem__(self, idx):
        job_description = self.data[idx]
        prompt = self.parent_data[idx]
        if job_description not in self.reward_cache:
            self.reward_cache[job_description] = self.reward_f(job_description, prompt) if self.reward_f is not None else 0.0
        return (prompt, job_description,), self.reward_cache[job_description] * self.reward_scale + self.reward_shift
    
    def __len__(self):
        return len(self.data)

