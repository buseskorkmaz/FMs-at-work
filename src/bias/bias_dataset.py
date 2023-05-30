import random
from typing import Optional
from data.rl_data import ConstantTokenReward, DataPoint, List_RL_Dataset, TokenReward
from bias.bias_base import StereosetData
from bias.bias_env import BiasObservation
from bias.bias_tokenizer import BiasTokenizer
import numpy as np
import random
import time

class BiasListDataset(List_RL_Dataset):
    def __init__(self, data: StereosetData, 
                 max_len: Optional[int], 
                 token_reward: TokenReward, 
                 cuttoff: Optional[float]=None, 
                 resample_timeout: float=0.0, 
                 include_parent: bool=True, 
                ) -> None:
        tokenizer = BiasTokenizer()
        super().__init__(tokenizer, token_reward, max_len)
        self.data = data
        self.cuttoff = cuttoff
        self.resample_timeout = resample_timeout
        self.include_parent = include_parent
    
    def get_item(self, idx: int):
        if self.cuttoff is not None:
            (parent, comment,), reward = random.choice(self.data)
            while reward < self.cuttoff:
                time.sleep(self.resample_timeout)
                (parent, comment,), reward = random.choice(self.data)
        else:
            (parent, comment,), reward = self.data[idx]
        obs = BiasObservation(parent if self.include_parent else None, comment, reward)
        return DataPoint.from_obs(obs, self.tokenizer, self.token_reward)
    
    def size(self):
        return len(self.data)
