from typing import List
import openai
from transformers import GPT2Tokenizer
import time
import random
from collections import defaultdict
from bias.bias_base import StereosetData
from bias.reward_model import RewardModel
import torch

def bias_reward():
    return get_bias_f(noise_1=False)

def bias_noised_reward():
    return get_bias_f(noise_1=True)

class HumanRewardScorer:
    def __init__(self, sentences_path: str, indexes: List[int]):
        data = StereosetData(sentences_path, indexes, None, None, 0.0, 1.0)
        self.data_index = defaultdict(list)
        for idx, ((_, c), _) in enumerate(data):
            self.data_index[c].append(data.gt_scores[idx])

    def score(self, text: str):
        if text in self.data_index:
            return 2 * float(random.choice(self.data_index[text]) > 0) - 1
        raise NotImplementedError

def score_human_reward(sentences_path: str, indexes: List[int]):
    scorer = HumanRewardScorer(sentences_path, indexes)
    return scorer.score

# def score_human_reward(sentences_path: str, indexes: List[int]):
#     data = StereosetData(sentences_path, indexes, None, None, 0.0, 1.0)
#     data_index = defaultdict(list)
#     for idx, ((_, c), _) in enumerate(data):
#         data_index[c].append(data.gt_scores[idx])
#     def _human_reward(text: str):
#         if text in data_index:
#             return 2 * float(random.choice(data_index[text]) > 0) - 1
#         raise NotImplementedError
#     return _human_reward

def model_reward(model: RewardModel):
    model.eval()
    def _model_reward(text: str):
        return model.get_reward_str(text)
    return _model_reward

def get_bias_f(noise_1: bool=False):
    print("I AM DOING WEIRD STUFF")


    # gpt3_tok = GPT2Tokenizer.from_pretrained('gpt2')
    # # reimplement it, we can't query openai
    
    # def _bias(text: str):
    #     print(text)

    #     input_text = parent_data  # Your input text
    #     input_ids = gpt3_tok.encode(input_text, return_tensors='pt')

    #     specific_word = text

    #     # Encode the input text and the specific word
    #     input_ids = gpt3_tok.encode(input_text, return_tensors='pt')
    #     word_ids = gpt3_tok.encode(specific_word)

    #     # Get the logits
    #     with torch.no_grad():
    #         outputs = model(input_ids)
    #         logits = outputs.logits

    #     # Get the logits of the specific word as the next token
    #     specific_word_logits = logits[0, -1, word_ids[0]]

    #     print(f"The logits of the word '{specific_word}' as the next token after the input '{input_text}' is {specific_word_logits.item()}")

    #     score = 

    #     return score
    raise NotImplementedError
