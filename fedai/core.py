"""Fill in a module description here"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_core.ipynb.

# %% auto 0
__all__ = ['say_hello', 'get_cfg', 'simple_model']

# %% ../nbs/00_core.ipynb 5
def say_hello(to):
    "Say hello to somebody"
    return f'Hello {to}!'
def get_cfg():
    from omegaconf import DictConfig, OmegaConf
    import argparse
    import os
    args = dict()
    args['dataset'] = 'dolly'
    args['model'] = "openai-community/gpt2"
    args['max_length'] = 1022
    args['num_clients'] = 80
    args['iid'] = "dir0.5"
    args['batch_size'] = 1
    args['dataset_subsample'] = 1.0
    args['eval_metric'] = 'loss'
    args['log_root'] = 'logs'
    args['save_dir'] = 'checkpoints'
    args['lora_alpha'] = 8
    args['lora_dropout'] = 0.05
    args['r'] = 32
    args['target_modules'] = ['c_attn']
    args['lr'] = 5e-5
    args['optimizer'] = 'Adam'
    args = argparse.Namespace(**args)
    args.name = 'mira'
    args.device = 0
    args.bias_sampling = False
    args.num_clients_per_task = int(args.num_clients/8)
    args.use_prompts= True
    args.rounds = 40
    args.m = 0.05
    cfg = OmegaConf.create(vars(args))
    return cfg



# %% ../nbs/00_core.ipynb 6
import torch
class simple_model(torch.nn.Module):
    def __init__(self):
        super(simple_model, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
