from typing import List
import random
from collections import defaultdict
from hackernews.hackernews_base import HackernewsData
# from hackernews.diversity_evaluator import Diversity_Evaluator
from load_objects_openllama import load_item

def hackernews_reward():
    return get_diversity_reward_f(noise_1=False)

def hackernews_noised_reward():
    return get_diversity_reward_f(noise_1=True)

class HumanRewardScorer:
    def __init__(self, indexes: List[int]):
        data = HackernewsData(indexes, None, None, 0.0, 1.0)
        print("Dataset:", data)
        self.data_index = defaultdict(list)
        for idx, ((parent, c), _) in enumerate(data):
            self.data_index[c].append(data.gt_scores[idx])

    def score(self, text: str, parent: str):
        if text in self.data_index:
            # print("\n",text, ":", self.data_index[text])
            return float(random.choice(self.data_index[text]))
        raise NotImplementedError

def score_human_reward(indexes: List[int]):
    scorer = HumanRewardScorer(indexes)
    return scorer.score

# def score_human_reward(job_descriptions_path: str, indexes: List[int]):
#     data = StereosetData(job_descriptions_path, indexes, None, None, 0.0, 1.0)
#     data_index = defaultdict(list)
#     for idx, ((_, c), _) in enumerate(data):
#         data_index[c].append(data.gt_scores[idx])
#     def _human_reward(text: str):
#         if text in data_index:
#             return 2 * float(random.choice(data_index[text]) > 0) - 1
#         raise NotImplementedError
#     return _human_reward

# def model_reward(model: RewardModel):
#     # var = 5
#     # model.eval(var)
#     def _model_reward(context:str, text: str):
#         return model.get_reward_str(context+text)
#     return _model_reward

def get_diversity_reward_f(noise_1: bool=False):
    # evaluator  = Diversity_Evaluator()
    config = {'diversity_reward': {'name': 'diversity_reward'}}
    evaluator = load_item(config['diversity_reward'])
    def _diversity(text: str, prompt:str):
        q_value= evaluator.calc_q_value(text, prompt)
        return q_value
    return _diversity
   