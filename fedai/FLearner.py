"""The entry point that defines all other steps"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/10_FLearner.ipynb.

# %% auto 0
__all__ = ['client_fn', 'FLearner']

# %% ../nbs/10_FLearner.ipynb 4
import os
from datetime import datetime
from omegaconf import OmegaConf
import argparse
import yaml
from copy import deepcopy
import torch
from torch import nn
from fastcore.utils import * 
from .federated.agents import * 
from .learner_utils import * 
from .client_selector import *  
from .core import get_cfg  
from .wandb_writer import *  

# %% ../nbs/10_FLearner.ipynb 5
def client_fn(client_cls, cfg, id, latest_round, t, loss_fn = None, optimizer = None, state_dir= None):
    
    model = get_model(cfg)
    criterion = get_criterion(loss_fn)
    train_block, test_block = get_block(cfg, id), get_block(cfg, id, train=False)

    state = {'model': model, 'optimizer': None, 'criterion': criterion, 't': t, 'h': None, 'h_c': None, "pers_model": None}

    if t == 1:
        state['w0'] = deepcopy(state['model'])
    
    if t == 1 and cfg.client_cls == "pFedMe" and cfg.agg  != "one_model":
        state = load_state_from_disk(cfg, state, latest_round, id, t, state_dir)  

        
    if t > 1:
        state = load_state_from_disk(cfg, state, latest_round, id, t, state_dir)  
        

    state['optimizer'] = get_cls("torch.optim", cfg.optimizer.name)(state['model'].parameters(), lr=cfg.lr)      
    state['alignment_criterion']= get_cls("torch.nn", cfg.alignment_criterion)
    
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
        cfg.root_dir = os.path.join(cfg.root_dir, cfg.project_name)
        self.cfg.save_dir = os.path.join(self.cfg.root_dir, self.cfg.now, self.cfg.save_dir)
        self.cfg.log_dir = os.path.join(self.cfg.root_dir, self.cfg.now, self.cfg.log_dir)
        self.cfg.res_dir = os.path.join(self.cfg.root_dir, self.cfg.now, self.cfg.res_dir)

        self.client_fn = client_fn
        
        self.client_selector = client_selector(self.cfg)
        self.client_cls = client_cls
        self.loss_fn = loss_fn()
        self.writer = writer(cfg)
        self.server  = self.client_cls(cfg= self.cfg, block= None, id= -1, state= None, role= AgentRole.SERVER)  # noqa: F405
        self.server.server_init(self.client_fn, self.client_selector, self.client_cls, self.loss_fn, self.writer)
    

# %% ../nbs/10_FLearner.ipynb 7
@patch
def run_simulation(self: FLearner):
    self.server.runFL()
