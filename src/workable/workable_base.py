from typing import Callable, List, Optional, Dict, Tuple
from utils.cache import Cache
from datasets import load_from_disk
from workable.workable_tokenizer import WorkableTokenizer
from data.rl_data import DataPoint
import torch
from data.language_environment import Language_Observation

class WorkableObservation(Language_Observation):
    def __init__(self, parent: Optional[str], text: Optional[str], reward: Optional[float]):
        assert (text is None and reward is None) or (text is not None and reward is not None)
        self.parent = parent
        self.text = text
        self.reward = reward
    
    def to_sequence(self) -> Tuple[List[Tuple[str, Optional[float]]], bool]:
        if self.text is None:
            if self.parent is not None:
                return [(self.parent, None)], False
            return [], False
        if self.parent is None:
            return [(self.text, self.reward)], True
        return [(self.parent, None), (self.text, self.reward)], True
    
    def __str__(self) -> str:
        if self.parent is not None:
            return f'parent: {self.parent}\ncomment: {self.text}'
        return self.text

class WorkableData:
    def __init__(self,
                 indexes: Optional[List[int]], 
                 reward_f: Optional[Callable[[str], float]], 
                 reward_cache: Optional[Cache]=None, 
                 reward_shift: float=0.0, 
                 reward_scale: float=1.0, 
                 prompt: Dict=None):
        
        if prompt is None:
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
        
        else:
            self.data = [prompt['description']]
            self.parent_data = [prompt['prompt']] 
            self.gt_scores = [-200]
            self.info = ("prompt", 1)
            self.reward_cache = None
            if self.reward_cache is None:
                self.reward_cache = Cache()

            self.reward_shift = reward_shift
            self.reward_scale = reward_scale
            print("INITIALIZED WITH REWARD F", reward_f)
            self.reward_f = reward_f
            self.max_len = 1024
            self.tokenizer = WorkableTokenizer()
            # self.token_reward = token_reward

    def __getitem__(self, idx):
        job_description = self.data[idx]
        prompt = self.parent_data[idx]
        if job_description not in self.reward_cache:
            self.reward_cache[job_description] = self.reward_f(job_description, prompt) if self.reward_f is not None else 0.0
        return (prompt, job_description,), self.reward_cache[job_description] * self.reward_scale + self.reward_shift

    def collate(self, p_items: List[DataPoint], device):
        items = []
        tokenizer = WorkableTokenizer()
        for item in p_items:
            parent_text_tuple = item[0]
            parent = parent_text_tuple[0]
            text = parent_text_tuple[1]
            reward = item[1]
            obs = WorkableObservation(parent=parent, text=text, reward=reward)
            it = DataPoint.from_obs(obs, tokenizer=tokenizer, token_reward=tokenizer.encode(text))
            items.append(it)

        tokens, state_idxs, action_idxs, rewards, terminals, u_state_idxs, u_action_idxs, u_rewards, u_terminals = zip(*map(lambda x: x.to_tensors(device, self.max_len), items))
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attn_mask = (tokens != self.tokenizer.pad_token_id).float()
        state_idxs = torch.nn.utils.rnn.pad_sequence(state_idxs, batch_first=True, padding_value=0)
        action_idxs = torch.nn.utils.rnn.pad_sequence(action_idxs, batch_first=True, padding_value=0)
        terminals = torch.nn.utils.rnn.pad_sequence(terminals, batch_first=True, padding_value=1)
        rewards = torch.nn.utils.rnn.pad_sequence(rewards, batch_first=True, padding_value=0.0)
        u_state_idxs = torch.nn.utils.rnn.pad_sequence(u_state_idxs, batch_first=True, padding_value=0)
        u_action_idxs = torch.nn.utils.rnn.pad_sequence(u_action_idxs, batch_first=True, padding_value=0)
        u_terminals = torch.nn.utils.rnn.pad_sequence(u_terminals, batch_first=True, padding_value=1)
        u_rewards = torch.nn.utils.rnn.pad_sequence(u_rewards, batch_first=True, padding_value=0.0)
        return {'tokens': tokens, 'attn_mask': attn_mask, 
                'state_idxs': state_idxs, 'action_idxs': action_idxs, 
                'rewards': rewards, 'terminals': terminals, 
                'u_state_idxs': u_state_idxs, 'u_action_idxs': u_action_idxs, 
                'u_rewards': u_rewards, 'u_terminals': u_terminals}

    def size(self):
        return self.__len__()

    def get_item(self, idx):
        return self.__getitem__(idx)

    def __len__(self):
        return len(self.data)

