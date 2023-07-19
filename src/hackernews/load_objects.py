from load_objects import *
import json
from hackernews.hackernews_base import HackernewsData
from hackernews.reward_fs import score_human_reward, hackernews_reward, hackernews_noised_reward
from hackernews.diversity_evaluator import Diversity_Evaluator
from hackernews.hackernews_dataset import HackernewsListDataset
from hackernews.hackernews_env import HackernewsEnvironment

@register('hackernews_reward')
def load_hackernews_reward(config, device, verbose=True):
    return hackernews_reward()

@register('hackernews_noised_reward')
def load_hackernews_noised_reward(config, device, verbose=True):
    return hackernews_noised_reward()

@register('score_human_reward')
def load_score_human_reward(config, device, verbose=True):
    if config['index_path'] is not None:
        with open(convert_path(config['index_path']), 'r') as f:
            indexes = json.load(f)
    else:
        indexes = None
    return score_human_reward(indexes)

@register('hackernews_rl_dataset')
def load_hackernews_rl_dataset(config, device, verbose=True):
    if config['reward_f'] is not None:
        reward_f = load_item(config['reward_f'], device, verbose=verbose)
    else:
        reward_f = None
    if config['index_path'] is not None:
        with open(convert_path(config['index_path']), 'r') as f:
            indexes = json.load(f)
    else:
        indexes = None
    data = HackernewsData(indexes, reward_f, None, 
                      config['reward_shift'], config['reward_scale'])
    if config['cache_path'] is not None:
        if verbose:
            print('loading hackernews reward cache from: %s' % convert_path(config['cache_path']))
        data.reward_cache.load(convert_path(config['cache_path']))
        if verbose:
            print('loaded.')
    return data

@register('hackernews_list_dataset')
def load_hackernews_list_dataset(config, device, verbose=True):
    data = load_item(config['data'], device, verbose=verbose)
    token_reward = load_item(config['token_reward'], device, verbose=verbose)
    return HackernewsListDataset(data, max_len=config['max_len'], 
                               token_reward=token_reward, 
                               cuttoff=None, 
                               resample_timeout=config['resample_timeout'], 
                               include_parent=True)

@register('hackernews_env')
def load_hackernews_env(config, device, verbose=True):
    data = load_item(config['data'], device, verbose=verbose)
    if config['reward_f'] is not None:
        reward_f = load_item(config['reward_f'], device, verbose=verbose)
    else:
        reward_f = None
    return HackernewsEnvironment(data=data, 
                               reward_f=reward_f, 
                               reward_shift=config['reward_shift'], 
                               reward_scale=config['reward_scale'], 
                               include_parent=config['include_parent'])

@register('diversity_reward')
def load_diversity_evaluator(config, verbose=True):
    return Diversity_Evaluator()

