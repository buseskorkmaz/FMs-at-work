from typing import Callable, List, Optional
from utils.cache import Cache
from datasets import load_from_disk

class WorkableData:
    def __init__(self,
                 indexes: Optional[List[int]], 
                 reward_f: Optional[Callable[[str], float]], 
                 reward_cache: Optional[Cache]=None, 
                 reward_shift: float=0.0, 
                 reward_scale: float=1.0):
        
        rl_dataset = load_from_disk("/dccstor/autofair/bias_llm/Bias-ILQL/data/workable_rl_dataset/job_descriptions_w_q_prompt_eng")
        items = [row for row in rl_dataset]
        print("Indexes:", indexes)
        if indexes is not None:
            items = [items[idx] for idx in indexes]
        self.info = ("huggingface", len(indexes))
        self.reward_cache = reward_cache
        if self.reward_cache is None:
            self.reward_cache = Cache()

        self.data = [item['description'] for item in items]
        self.parent_data = [item['prompt'] for item in items]
        self.gt_scores = [float(item['q_value']) for item in items]
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

