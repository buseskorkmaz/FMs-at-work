from typing import List
import openai
from transformers import GPT2Tokenizer
import time
import random
from collections import defaultdict
from bias.bias_base import StereosetData
from bias.reward_model import RewardModel
import torch
import os

def bias_reward():
    return get_bias_f(noise_1=False)

def bias_noised_reward():
    return get_bias_f(noise_1=True)

class HumanRewardScorer:
    def __init__(self, sentences_path: str, indexes: List[int]):
        data = StereosetData(sentences_path, indexes, None, None, 0.0, 1.0)
        self.data_index = defaultdict(list)
        for idx, ((parent, c), _) in enumerate(data):
            self.data_index[parent+c].append(data.gt_scores[idx])

    def score(self, context:str, text: str):
        text = context+text
        if text in self.data_index:
            # print("\n",text, ":", self.data_index[text])
            return float(random.choice(self.data_index[text]))
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
    # var = 5
    # model.eval(var)
    def _model_reward(context:str, text: str):
        return model.get_reward_str(context+text)
    return _model_reward

def get_bias_f(noise_1: bool=False):
    print("I AM DOING WEIRD STUFF")

    gpt3_tok = GPT2Tokenizer.from_pretrained('gpt2')
    def _bias(context:str, text: str):
        text = context + text
        # only take the first 2042 tokens to prevent context length errors
        text = gpt3_tok.decode(gpt3_tok(text, add_special_tokens=False)['input_ids'][:20], clean_up_tokenization_spaces=False)
        while True:
            try:
                # response = openai.Completion.create(
                # engine="content-filter-alpha",
                # prompt = "<|endoftext|>"+text+"\n--\nLabel:",
                # temperature=0,
                # max_tokens=1,
                # top_p=0,
                # logprobs=10, 
                # )
                openai.api_key = os.getenv("OPENAI_API_KEY")
                response = openai.Moderation.create(
                    input=text,
                )
                print("\n"+text+"\n")
                print(response)
                break
            except:
                time.sleep(1)
                continue
        output_label = sum(response["results"][0]["category_scores"].values())
        # toxic_threshold = -0.355
        # if output_label == "2":
        #     logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]
        #     if logprobs["2"] < toxic_threshold:
        #         logprob_0 = logprobs.get("0", None)
        #         logprob_1 = logprobs.get("1", None)
        #         if logprob_0 is not None and logprob_1 is not None:
        #             if logprob_0 >= logprob_1:
        #                 output_label = "0"
        #             else:
        #                 output_label = "1"
        #         elif logprob_0 is not None:
        #             output_label = "0"
        #         elif logprob_1 is not None:
        #             output_label = "1"
        # if output_label not in ["0", "1", "2"]:
        #     output_label = "2"
        # if output_label == "1" and noise_1:
        #     output_label = random.choice(["0", "2"])
        if output_label > 0.3:
            output_label = 1
        else:
            output_label = 0
        
        if output_label == 1 and noise_1:
            output_label = random.choice([1, 2])

        return output_label
    return _bias
    # TAKE IT FROM TOXICITY
    # raise NotImplementedError
