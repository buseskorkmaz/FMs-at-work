from typing import Callable, List, Optional
from utils.cache import Cache
import json

class HackernewsData:
    def __init__(self,
                 indexes: Optional[List[int]], 
                 reward_f: Optional[Callable[[str], float]], 
                 reward_cache: Optional[Cache]=None, 
                 reward_shift: float=0.0, 
                 reward_scale: float=1.0):
        
        with open('/dccstor/autofair/busekorkmaz/FMs-at-work/data/hackernews_rl_dataset/prompts.json', 'r') as file:
            rl_dataset = json.load(file)

        items = [row for row in rl_dataset]
        self.reward_cache = reward_cache
        if self.reward_cache is None:
            self.reward_cache = Cache()

        self.data = [item['cleaned_text'] for item in items]
        self.parent_data = [item['prompt'] for item in items]
        self.gt_scores = [-100 if float(item['q_val']) == -1000.0 else float(item['q_val']) for item in items]
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale
        self.reward_f = reward_f


    def __getitem__(self, idx):
        job_description = self.data[idx]
        prompt = self.parent_data[idx]
        if job_description not in self.reward_cache:
            self.reward_cache[job_description] = self.reward_f(job_description, prompt) if self.reward_f is not None else 0.0
        return (prompt, job_description,), self.reward_cache[job_description] * self.reward_scale + self.reward_shift
    
    def __len__(self):
        return len(self.data)

