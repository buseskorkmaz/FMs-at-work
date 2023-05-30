from src.load_objects import *
import json
from bias.bias_base import StereosetData
from bias.reward_fs import model_reward, score_human_reward, bias_reward, bias_noised_reward
from bias.reward_model import RobertaBinaryRewardModel
from bias.bias_dataset import BiasListDataset
from bias.bias_env import BiasEnvironment

@register('bias_reward')
def load_bias_reward(config, device, verbose=True):
    return bias_reward()

@register('bias_noised_reward')
def load_bias_noised_reward(config, device, verbose=True):
    return bias_noised_reward()

@register('score_human_reward')
def load_score_human_reward(config, device, verbose=True):
    if config['index_path'] is not None:
        with open(convert_path(config['index_path']), 'r') as f:
            indexes = json.load(f)
    else:
        indexes = None
    return score_human_reward(convert_path(config['sentences_path']), indexes)

@register('model_reward')
def load_model_reward(config, device, verbose=True):
    model = load_item(config['model'], device, verbose=verbose)
    return model_reward(model)

@register('bias_sentences')
def load_bias_sentences(config, device, verbose=True):
    if config['reward_f'] is not None:
        reward_f = load_item(config['reward_f'], device, verbose=verbose)
    else:
        reward_f = None
    if config['index_path'] is not None:
        with open(convert_path(config['index_path']), 'r') as f:
            indexes = json.load(f)
    else:
        indexes = None
    data = StereosetData(convert_path(config['path']), indexes, reward_f, None, 
                      config['reward_shift'], config['reward_scale'])
    if config['cache_path'] is not None:
        if verbose:
            print('loading bias reward cache from: %s' % convert_path(config['cache_path']))
        data.reward_cache.load(convert_path(config['cache_path']))
        if verbose:
            print('loaded.')
    return data

@register('bias_list_dataset')
def load_bias_list_dataset(config, device, verbose=True):
    data = load_item(config['data'], device, verbose=verbose)
    token_reward = load_item(config['token_reward'], device, verbose=verbose)
    return BiasListDataset(data, max_len=config['max_len'], 
                               token_reward=token_reward, 
                               cuttoff=config['cuttoff'], 
                               resample_timeout=config['resample_timeout'], 
                               include_parent=True)

@register('bias_env')
def load_bias_env(config, device, verbose=True):
    data = load_item(config['data'], device, verbose=verbose)
    if config['reward_f'] is not None:
        reward_f = load_item(config['reward_f'], device, verbose=verbose)
    else:
        reward_f = None
    return BiasEnvironment(data=data, 
                               reward_f=reward_f, 
                               reward_shift=config['reward_shift'], 
                               reward_scale=config['reward_scale'], 
                               include_parent=config['include_parent'])

@register('roberta_binary_reward_model')
def load_roberta_binary_reward_model(config, device, verbose=True):
    dataset = load_item(config['dataset'], device, verbose=verbose)
    model = RobertaBinaryRewardModel(dataset, device, config['roberta_kind'], 
                                     freeze_roberta=config['freeze_roberta'], 
                                     reward_cuttoff=config['reward_cuttoff'])
    return load_model(config['load'], model, device, verbose=verbose)

