import json
from models.bc_lm_bloom import BC_LM, BC_Evaluator, BC_Policy
import torch
from models.bcq_model import BCQModel, GModel, PsiModel
from models.cql_model import CQLModel
from models.dt_model import DT, DT_Evaluator
from models.iql_model_bloom import IQL_Evaluator, IQL_Policy, PerTokenIQL, TopAdvantageNGrams
from models.utterance_iql_model import PerUtteranceIQL, PerUtteranceIQL_Policy, UtteranceIQL_Evaluator
from data.rl_data import ConstantTokenReward, SepcifiedTokenReward
from models.chai_model import Chai_Evaluator, ChaiModel, ChaiPolicy
from utils.cache import Cache
from utils.misc import convert_path
# from models.gpt2_optional_final_ln import GPT2LMHeadModel, GPT2Config, GPT2Model
from transformers import PreTrainedModel, BloomForCausalLM, BloomConfig

registry = {}
cache = {}

def register(name):
    def add_f(f):
        registry[name] = f
        return f
    return add_f

def load_item(config, *args, verbose=True):
    config = config.copy()
    print("Config: ", config)
    name = config.pop('name')
    print("Registry:",registry)
    print(name)
    if name not in registry:
        raise NotImplementedError
    if 'cache_id' in config:
        if (name, config['cache_id']) in cache:
            if verbose:
                print(f'loading from cache ({name}, {config["cache_id"]})')
            return cache[(name, config['cache_id'])]
    if verbose:
        print(f'loading {name}: {config}')
    item = registry[name](config, *args, verbose=verbose)
    if 'cache_id' in config:
        print(f'saving to cache ({name}, {config["cache_id"]})')
        cache[(name, config['cache_id'])] = item
    return item

def load_model(config, model, device, verbose=True):
    model = model.to(device)
    if config['checkpoint_path'] is not None:
        if verbose:
            print('loading %s state dict from: %s' % (config['name'], convert_path(config["checkpoint_path"])))
        model.load_state_dict(torch.load(convert_path(config['checkpoint_path']), map_location='cuda'), strict=config['strict_load'])
        if verbose:
            print('loaded.')
    return model

@register('constant_token_reward')
def load_constant_token_reward(config, device, verbose=True):
    return ConstantTokenReward(config['c'])

@register('specified_token_reward')
def load_specified_token_reward(config, device, verbose=True):
    with open(convert_path(config['token_file']), 'r') as f:
        token_data = {int(k): v for k, v in json.load(f).items()}
    return SepcifiedTokenReward(token_data, config['scale'], config['shift'])

@register('bloom3b')
def load_bloom(config, verbose=True):
    obj = BloomForCausalLM if config['lm_head'] else PreTrainedModel
    if config['from_pretrained']:
        model = obj.from_pretrained(config['bloom3b_type'])
        print(f"--> ModelBloom")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> Bloom has {total_params / 1e6} Million params\n")
        # model = prepare_model_for_int8_training(model)
        # peft_config = LoraConfig(
        #     lora_alpha=16,
        #     lora_dropout=0.1,
        #     r=64,
        #     bias="none",
        #     task_type="CAUSAL_LM",
        #     target_modules = [
        #         "q_proj",
        #         "up_proj",
        #         "o_proj",
        #         "k_proj",
        #         "down_proj",
        #         "gate_proj",
        #         "v_proj"
        #     ],
        # )
        # model = get_peft_model(model, peft_config)
        # model.print_trainable_parameters()
        return model
    config = BloomConfig.from_pretrained(config['bloom3b_type'])
    return obj(config)

@register('bc_lm')
def load_bc_lm(config, device, verbose=True):
    bloom = load_item(config['bloom3b'], verbose=verbose)
    dataset = load_item(config['dataset'], device, verbose=verbose)
    model = BC_LM(bloom, dataset, device, config['transition_weight'])
    return load_model(config['load'], model, device, verbose=verbose)

@register('bc_policy')
def load_bc_policy(config, device, verbose=True):
    bc_lm = load_item(config['bc_lm'], device, verbose=verbose)
    return BC_Policy(bc_lm, config['kind'], **config['generation_kwargs'])

@register('bc_evaluator')
def load_bc_evaluator(config, device, verbose=True):
    env = load_item(config['env'], device, verbose=verbose)
    return BC_Evaluator(env, config['env'], config['kind'], **config['generation_kwargs'])

@register('per_token_iql_bloom')
def load_per_token_iql_bloom(config, device, verbose=True):
    bloom = load_item(config['bloom3b'], verbose=verbose)
    dataset = load_item(config['dataset'], device, verbose=verbose)
    model = PerTokenIQL(bloom, dataset, device, config['alpha'], config['gamma'], 
                        config['beta'], config['transition_weight'], config['clip_weight'], 
                        config['value_max'], config['value_min'], config['detach_v'], 
                        config['detach_pi'], config['detach_q'], config['double_q'], 
                        config['tau'], config['seperate_policy'], config['seperate_target'], 
                        config['exp_weights'], config['dm_margin'], config['advanced_mlp'], 
                        config['cql_temp'])
    return load_model(config['load'], model, device, verbose=verbose)

@register('per_token_cql')
def load_per_token_cql(config, device, verbose=True):
    bloom = load_item(config['bloom3b'], verbose=verbose)
    dataset = load_item(config['dataset'], device, verbose=verbose)
    model = CQLModel(bloom, dataset, device, config['alpha'], config['gamma'], 
                     config['beta'], config['transition_weight'], config['clip_weight'], 
                     config['value_max'], config['value_min'], config['detach_v'], 
                     config['detach_pi'], config['detach_q'], config['double_q'], 
                     config['seperate_policy'], config['seperate_target'], config['exp_weights'], 
                     config['advanced_mlp'], config['cql_temp'])
    return load_model(config['load'], model, device, verbose=verbose)

@register('per_token_bcq')
def load_per_token_bcq(config, device, verbose=True):
    bloom = load_item(config['bloom3b'], verbose=verbose)
    dataset = load_item(config['dataset'], device, verbose=verbose)
    model = BCQModel(bloom, dataset, device, config['alpha'], config['gamma'], 
                     config['beta'], config['transition_weight'], config['clip_weight'], 
                     config['value_max'], config['value_min'], config['detach_v'], 
                     config['detach_pi'], config['detach_q'], config['double_q'], 
                     config['seperate_policy'], config['seperate_target'], config['exp_weights'], 
                     config['advanced_mlp'], config['cql_temp'])
    return load_model(config['load'], model, device, verbose=verbose)

@register('per_token_psi')
def load_per_token_psi(config, device, verbose=True):
    bloom = load_item(config['bloom3b'], verbose=verbose)
    dataset = load_item(config['dataset'], device, verbose=verbose)
    model = PsiModel(bloom, dataset, device, config['alpha'], config['gamma'], 
                     config['beta'], config['transition_weight'], config['clip_weight'], 
                     config['value_max'], config['value_min'], config['detach_v'], 
                     config['detach_pi'], config['detach_q'], config['double_q'], 
                     config['seperate_policy'], config['seperate_target'], config['exp_weights'], 
                     config['advanced_mlp'], config['cql_temp'])
    return load_model(config['load'], model, device, verbose=verbose)

@register('per_token_g')
def load_per_token_g(config, device, verbose=True):
    bloom = load_item(config['bloom3b'], verbose=verbose)
    dataset = load_item(config['dataset'], device, verbose=verbose)
    model = GModel(bloom, dataset, device, config['alpha'], config['gamma'], 
                   config['beta'], config['transition_weight'], config['clip_weight'], 
                   config['value_max'], config['value_min'], config['detach_v'], 
                   config['detach_pi'], config['detach_q'], config['double_q'], 
                   config['seperate_policy'], config['seperate_target'], config['exp_weights'], 
                   config['advanced_mlp'], config['cql_temp'])
    return load_model(config['load'], model, device, verbose=verbose)

@register('per_utterance_iql')
def load_per_utterance_iql(config, device, verbose=True):
    bloom = load_item(config['bloom3b'], verbose=verbose)
    dataset = load_item(config['dataset'], device, verbose=verbose)
    model = PerUtteranceIQL(bloom, dataset, device, config['alpha'], config['gamma'], 
                            config['beta'], config['transition_weight'], config['clip_weight'], 
                            config['value_max'], config['value_min'], config['detach_v'], 
                            config['detach_pi'], config['detach_q'], config['double_q'], 
                            config['tau'], config['seperate_policy'], config['seperate_target'], 
                            config['exp_weights'], config['advanced_mlp'])
    return load_model(config['load'], model, device, verbose=verbose)

@register('chai_model')
def load_chai_model(config, device, verbose=True):
    bloom = load_item(config['bloom3b'], verbose=verbose)
    dataset = load_item(config['dataset'], device, verbose=verbose)
    if config['use_cache']:
        cache = Cache()
        if config['cache_path'] is not None:
            if verbose:
                print('loading generation cache from: %s' % convert_path(config['cache_path']))
            cache.load(convert_path(config['cache_path']))
            if verbose:
                print('loaded.')
    else:
        cache = None
    model = ChaiModel(dataset, device, bloom, config['alpha'], config['gamma'], cache)
    return load_model(config['load'], model, device, verbose=verbose)

@register('iql_policy')
def load_iql_policy(config, device, verbose=True):
    iql_model = load_item(config['iql_model'], device, verbose=verbose)
    return IQL_Policy(iql_model, config['kind'], **config['generation_kwargs'])

@register('utterance_iql_policy')
def load_utterance_iql_policy(config, device, verbose=True):
    iql_model = load_item(config['iql_model'], device, verbose=verbose)
    return PerUtteranceIQL_Policy(iql_model, config['kind'], **config['generation_kwargs'])

@register('chai_policy')
def load_chai_policy(config, device, verbose=True):
    chai_model = load_item(config['chai_model'], device, verbose=verbose)
    return ChaiPolicy(chai_model, **config['generation_kwargs'])

@register('iql_evaluator')
def load_iql_evaluator(config, device, verbose=True):
    env = load_item(config['env'], device, verbose=verbose)
    return IQL_Evaluator(env, config['verbose'], config['kind'], **config['generation_kwargs'])

@register('top_advantage_n_grams')
def load_top_advantage_n_grams(config, device, verbose=True):
    data = load_item(config['data'], device, verbose=verbose)
    return TopAdvantageNGrams(data, config['print_every'], config['print_k'], config['n_gram'])

@register('utterance_iql_evaluator')
def load_utterance_iql_evaluator(config, device, verbose=True):
    env = load_item(config['env'], device, verbose=verbose)
    return UtteranceIQL_Evaluator(env, config['verbose'], config['kind'], **config['generation_kwargs'])

@register('chai_evaluator')
def load_chai_evaluator(config, device, verbose=True):
    env = load_item(config['env'], device, verbose=verbose)
    return Chai_Evaluator(env, config['verbose'], convert_path(config['cache_save_path']), **config['generation_kwargs'])

@register('dt_model')
def load_dt_model(config, device, verbose=True):
    bloom = load_item(config['bloom3b'], verbose=verbose)
    dataset = load_item(config['dataset'], device, verbose=verbose)
    model = DT(bloom, dataset, device, config['transition_weight'])
    return load_model(config['load'], model, device, verbose=verbose)

@register('dt_evaluator')
def load_dt_evaluator(config, device, verbose=True):
    env = load_item(config['env'], device, verbose=verbose)
    return DT_Evaluator(env, config['verbose'], config['kind'], **config['generation_kwargs'])
