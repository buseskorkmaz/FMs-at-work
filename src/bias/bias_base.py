import os
import csv
from typing import Callable, List, Optional
from src.utils.cache import Cache

class StereosetData:
    def __init__(self, path: str, 
                 indexes: Optional[List[int]], 
                 reward_f: Optional[Callable[[str], float]], 
                 reward_cache: Optional[Cache]=None, 
                 reward_shift: float=0.0, 
                # not sure about them
                 reward_scale: float=1.0):
        with open(os.path.join(path, 'stereoset.csv'), 'r') as f:
            items = [row for row in csv.reader(f)][1:]

        # TODO: check if rest of this ok
        if indexes is not None:
            items = [items[idx] for idx in indexes]
        self.info = (path, len(indexes))
        self.reward_cache = reward_cache
        if self.reward_cache is None:
            self.reward_cache = Cache()
        self.data = [item[1] for item in items]
        self.parent_data = [item[0] for item in items]
        self.gt_scores = [int(item[2]) for item in items]
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale
        self.reward_f = reward_f

    def __getitem__(self, idx):
        sentence = self.data[idx]
        context = self.parent_data[idx]
        if sentence not in self.reward_cache:
            self.reward_cache[sentence] = self.reward_f(sentence) if self.reward_f is not None else 0.0
        return (context, sentence,), self.reward_cache[sentence] * self.reward_scale + self.reward_shift
    
    def __len__(self):
        return len(self.data)

