import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../src'))
from torch.utils.data.dataset import IterableDataset
from data.rl_data import Iterable_RL_Dataset
from data.torch_datasets import GeneralDataset, GeneralIterDataset
from hackernews.load_objects import load_item
from accelerate import Accelerator
from utils.misc import add_system_configs, convert_path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import random
from omegaconf import DictConfig
from src.models.iql_model import IQL_Policy
from src.data.language_environment import interact_environment
from src.hackernews.hackernews_env import HackernewsData, HackernewsEnvironment

def generations(cfg : DictConfig):
    eval_cfg = cfg['eval']
    accelerator = Accelerator()
    system_cfg = add_system_configs(cfg, accelerator)
    print('using device:', system_cfg['device'])
    print('num processes:', system_cfg['num_processes'])
    print('using fp16:', system_cfg['use_fp16'])
    if eval_cfg['seed'] is not None:
        random.seed(eval_cfg['seed']+(torch.cuda.current_device() if torch.cuda.is_available() else 0))
    
    raw_dataset = load_item(cfg['dataset'], system_cfg['device'], verbose=False)
    if isinstance(raw_dataset, Iterable_RL_Dataset):
        dataset = GeneralIterDataset(raw_dataset, 'cpu')
    else:
        dataset = GeneralDataset(raw_dataset, 'cpu')
        
    data_loader_kwargs = {'num_workers': eval_cfg['dataloader_workers'], 
                            'batch_size': eval_cfg['bsize'], 
                            'collate_fn': dataset.collate}
    if not isinstance(dataset, IterableDataset):
        data_loader_kwargs['shuffle'] = False
    data_loader = DataLoader(dataset, **data_loader_kwargs)

    evaluator = None
    if cfg['evaluator'] is not None:
        evaluator = load_item(cfg['evaluator'], system_cfg['device'], verbose=False)

    model = load_item(cfg['model'], system_cfg['device'], verbose=False)

    model = accelerator.prepare(model)
    model.eval()
    data = HackernewsData(indexes=None, reward_f=None)
    env = HackernewsEnvironment(data, reward_f=None)

    kind = "sample"
    generation_kwargs = cfg['evaluator']['generation_kwargs']
    policy = IQL_Policy(accelerator.unwrap_model(model), kind, **generation_kwargs)
    with torch.no_grad():
        for i, item in tqdm(enumerate(data_loader)):
            result, sequence = interact_environment(env, policy, None)
            print("IQL GENERATIONS:")
            print(result)
            print('='*50)
            print("latest logits")
            print(policy.logits_all[-1])