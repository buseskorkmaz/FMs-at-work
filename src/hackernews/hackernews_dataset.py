import random
from typing import Optional
from data.rl_data import ConstantTokenReward, DataPoint, List_RL_Dataset, TokenReward
from hackernews.hackernews_base import HackernewsData
from hackernews.hackernews_env import HackernewsObservation
from hackernews.hackernews_tokenizer_openllama import HackernewsTokenizer
import numpy as np
import random
import time

class HackernewsListDataset(List_RL_Dataset):
    def __init__(self, data: HackernewsData, 
                 max_len: Optional[int], 
                 token_reward: TokenReward, 
                 cuttoff: Optional[float]=None, 
                 resample_timeout: float=0.0, 
                 include_parent: bool=True, 
                ) -> None:
        tokenizer = HackernewsTokenizer()
        super().__init__(tokenizer, token_reward, max_len)
        self.data = data
        self.cuttoff = cuttoff
        self.resample_timeout = resample_timeout
        self.include_parent = include_parent
    
    def get_item(self, idx: int):
        if self.cuttoff is not None:
            print("I AM DOING WWWEIRD STUFF")
            (parent, comment,), reward = random.choice(self.data)
            while reward < self.cuttoff:
                time.sleep(self.resample_timeout)
                (parent, comment,), reward = random.choice(self.data)
        else:
            # print("I AM NORMAL")
            (parent, comment,), reward = self.data[idx]
        # print("parent", parent)
        # print("comment", comment)
        # print("reward", reward)
        obs = HackernewsObservation(parent if self.include_parent else None, comment, reward)
        # print("Observation",obs.__str__())
        # print(comment+reward)
        return DataPoint.from_obs(obs, self.tokenizer, self.token_reward)
    
    def size(self):
        return len(self.data)
