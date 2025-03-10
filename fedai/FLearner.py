"""Fill in a module description here"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/10_FLearner.ipynb.

# %% auto 0
__all__ = ['client_fn', 'FLearner']

# %% ../nbs/10_FLearner.ipynb 4
import torch
import os
from datetime import datetime
from fastcore.utils import * # type: ignore # noqa: F403
from .federated.agents import * # noqa: F403
from .learner_utils import * # type: ignore # noqa: F403
from .client_selector import *  # noqa: F403
from .core import get_cfg  # noqa: F401, F403
from .wandb_writer import *  # noqa: F403
from torch import nn
from omegaconf import OmegaConf
import argparse
import yaml

# %% ../nbs/10_FLearner.ipynb 5
def client_fn(client_cls, cfg, id, latest_round, t, loss_fn = None, optimizer = None):
    
    model = get_model(cfg)
    criterion = get_criterion(loss_fn)

    train_block = get_block(cfg, id)
    test_block = get_block(cfg, id, train=False)    
    
    state = {'model': model, 'optimizer': None, 'criterion': criterion, 't': t}
    
    if t > 1:
        state = load_state_from_disk(cfg, state, latest_round, id, t)  # noqa: F405
    
    state['optimizer'] = get_cls("torch.optim", cfg.optimizer)(state['model'].parameters(), lr=cfg.lr)
        
    return client_cls(id, cfg, state, block= [train_block, test_block])


# %% ../nbs/10_FLearner.ipynb 6
class FLearner:
    def __init__(self,
                 cfg, # OmegaConf object
                 client_fn, # a function that returns a client object
                 client_selector= BaseClientSelector, # a client selection class represnting a client seleection algorithm # noqa: F405
                 client_cls= FLAgent,  # noqa: F405
                 loss_fn= torch.nn.CrossEntropyLoss,  # noqa: F405
                 writer= WandbWriter): # a writer to write results to an expirement tracking tool # noqa: F405
        
        self.cfg = cfg
        self.cfg.now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cfg.save_dir = os.path.join(self.cfg.project_name, self.cfg.now, self.cfg.save_dir)
        self.cfg.log_dir = os.path.join(self.cfg.project_name, self.cfg.now, self.cfg.log_dir)
        self.cfg.res_dir = os.path.join(self.cfg.project_name, self.cfg.now, self.cfg.res_dir)

        self.client_fn = client_fn
        self.server  = self.client_cls(cfg= self.cfg, block= None, id= -1, state= None, role= AgentRole.SERVER)  # noqa: F405

        self.client_selector = client_selector(self.cfg)
        self.client_cls = client_cls
        self.loss_fn = loss_fn()
        self.writer = writer(cfg)
        self.server.server_init(self.cfg, self.client_fn, self.client_cls, self.loss_fn, self.writer)
    

# %% ../nbs/10_FLearner.ipynb 7
@patch
def run_simulation(self: FLearner):
    self.server.runFL()
