import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../src'))
from torch.utils.data.dataset import IterableDataset
from data.rl_data import Iterable_RL_Dataset
from data.torch_datasets import GeneralDataset, GeneralIterDataset
from workable.load_objects import load_item
from accelerate import Accelerator
from utils.log_utils import DistributeCombineLogs, label_logs
from utils.misc import add_system_configs, convert_path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from functools import partial
from utils.torch_utils import to
import random
import pickle as pkl

def eval(cfg, prompt):
    print('using config:', cfg)
    eval_cfg = cfg['eval']
    accelerator = Accelerator()
    system_cfg = add_system_configs(cfg, accelerator)
    print('using device:', system_cfg['device'])
    print('num processes:', system_cfg['num_processes'])
    print('using fp16:', system_cfg['use_fp16'])
    if eval_cfg['seed'] is not None:
        random.seed(eval_cfg['seed']+(torch.cuda.current_device() if torch.cuda.is_available() else 0))
        # random.seed(eval_cfg['seed'])
    
    # wrapping prompt to dataset instance
    prompt_data = {"description": "",
                   "prompt": prompt,
                   "q_value": None}

    args = {"device": system_cfg['device'], "prompt": prompt_data}
    print("Dataset config:", cfg['dataset'])
    raw_dataset = load_item(cfg['dataset'], args)
    print(isinstance(raw_dataset, Iterable_RL_Dataset))

    if isinstance(raw_dataset, Iterable_RL_Dataset):
        dataset = GeneralIterDataset(raw_dataset, 'cpu')
    else:
        dataset = GeneralDataset(raw_dataset, 'cpu')
    print(dataset)
    data_loader_kwargs = {'num_workers': eval_cfg['dataloader_workers'], 
                          'batch_size': 1, 
                          'collate_fn': dataset.collate
                        }
    if not isinstance(dataset, IterableDataset):
        data_loader_kwargs['shuffle'] = False
    data_loader = DataLoader(dataset, **data_loader_kwargs)

    evaluator = None
    if cfg['evaluator'] is not None:
        args = {
            "device": system_cfg['device'],
            "prompt": prompt_data
        }
        evaluator = load_item(cfg['evaluator'], args)

    model = load_item(cfg['model'], system_cfg['device'])

    # print("Dataset length", len(dataset))
    # print("Dataset0", dataset[0])

    if isinstance(dataset, IterableDataset):
        model = accelerator.prepare(model)
    else:
        model, data_loader = accelerator.prepare(model, data_loader)
    model.eval()

    data_loader_kwargs = {'num_workers': eval_cfg['dataloader_workers'], 
                          'batch_size': 1, 
                          'collate_fn': dataset.collate
                          }
    if not isinstance(dataset, IterableDataset):
        data_loader_kwargs['shuffle'] = False
    data_loader = DataLoader(dataset, **data_loader_kwargs)
    eval_logs = DistributeCombineLogs(accelerator, use_wandb=False)
    with torch.no_grad():
        # for i, eval_items in tqdm(enumerate(data_loader)):
        #     eval_items = to(eval_items, system_cfg['device'])
        #     if i >= eval_cfg['batches']:
        #         break
        #     if evaluator is not None:
        #         evaluator.inference(accelerator.unwrap_model(model), eval_items)
        #         # print(evaluator_logs)
        eval_items = next(iter(data_loader))  # Fetch the first item directly
        eval_items = to(eval_items, system_cfg['device'])
        if evaluator is not None:
            evaluator.inference(accelerator.unwrap_model(model), eval_items)