from load_objects import *
import json
from workable.workable_base import WorkableData
from workable.reward_fs import score_human_reward, workable_reward, workable_noised_reward
from workable.diversity_evaluator import Diversity_Evaluator
from workable.workable_dataset import WorkableListDataset, WorkablePromptDataset
from workable.workable_env import WorkableEnvironment

@register('workable_reward')
def load_workable_reward(config, device, verbose=True):
    return workable_reward()

@register('workable_noised_reward')
def load_workable_noised_reward(config, device, verbose=True):
    return workable_noised_reward()

@register('score_human_reward')
def load_score_human_reward(config, device, verbose=True):
    if config['index_path'] is not None:
        with open(convert_path(config['index_path']), 'r') as f:
            indexes = json.load(f)
    else:
        indexes = None
    return score_human_reward(indexes)

@register('workable_rl_dataset')
def load_workable_rl_dataset(config, device, verbose=True):
    if config['reward_f'] is not None:
        reward_f = load_item(config['reward_f'], device, verbose=verbose)
    else:
        reward_f = None
    if config['index_path'] is not None:
        with open(convert_path(config['index_path']), 'r') as f:
            indexes = json.load(f)
    else:
        indexes = None
    data = WorkableData(indexes, reward_f, None, 
                      config['reward_shift'], config['reward_scale'])
    if config['cache_path'] is not None:
        if verbose:
            print('loading workable reward cache from: %s' % convert_path(config['cache_path']))
        data.reward_cache.load(convert_path(config['cache_path']))
        if verbose:
            print('loaded.')
    return data

@register('workable_list_dataset')
def load_workable_list_dataset(config, device, verbose=True):
    data = load_item(config['data'], device, verbose=verbose)
    token_reward = load_item(config['token_reward'], device, verbose=verbose)
    return WorkableListDataset(data, max_len=config['max_len'], 
                               token_reward=token_reward, 
                               cuttoff=None, 
                               resample_timeout=config['resample_timeout'], 
                               include_parent=True)

@register('workable_env')
def load_workable_env(config, args, verbose=True):
    device= args["device"] 
    prompt= args["prompt"]
    print("Prompt:", prompt)
    if prompt is None:
        data = load_item(config['data'], device, verbose=verbose)
        if config['reward_f'] is not None:
            reward_f = load_item(config['reward_f'], device, verbose=verbose)
        else:
            reward_f = None
    
    else:
        data = load_item(config['data'], args, verbose=verbose)
        print("Loading environment with data", data)
        if config['reward_f'] is not None:
            reward_f = load_item(config['reward_f'], device, verbose=verbose)
        else:
            reward_f = None

    return WorkableEnvironment(data=data, 
                            reward_f=reward_f, 
                            reward_shift=config['reward_shift'], 
                            reward_scale=config['reward_scale'], 
                            include_parent=config['include_parent'])
    

@register('diversity_reward')
def load_diversity_evaluator(config, verbose=True):
    return Diversity_Evaluator()

@register('prompt_dataset')
def load_workable_prompt_dataset(config, args, verbose=True):
    device= args["device"]
    prompt = args["prompt"]
    print("CONFIG", config)
    # print("prompt:", prompt)
    # print(device)
    try:
        if config['data']['reward_f'] is not None:
            reward_f = load_item(config['data']['reward_f'], device, verbose=verbose)
        else:
            reward_f = None
    except:
        if config['reward_f'] is not None:
            reward_f = load_item(config['reward_f'], device, verbose=verbose)
        else:
            reward_f = None
    # if config['index_path'] is not None:
    #     with open(convert_path(config['index_path']), 'r') as f:
    #         indexes = json.load(f)
    # else:
    indexes = None
    reward_shift = 0.0
    reward_scale = 1.0
    data = WorkableData(indexes, reward_f, None, 
                      reward_shift,reward_scale, prompt=prompt)
    if 'token_reward' in config.keys():
        token_reward = load_item(config['token_reward'], device, verbose=verbose)
    else:
        print("skipping token reward")
    # if config['cache_path'] is not None:
    #     if verbose:
    #         print('loading workable reward cache from: %s' % convert_path(config['cache_path']))
    #     data.reward_cache.load(convert_path(config['cache_path']))
    #     if verbose:
    #         print('loaded.')

    return WorkablePromptDataset(data=data, max_len=config['max_len'], 
                               token_reward=token_reward, 
                               cuttoff=None, 
                               resample_timeout=config['resample_timeout'], 
                               include_parent=True)
