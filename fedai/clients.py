"""The core abstraction for different FL Clients."""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_clients.ipynb.

# %% auto 0
__all__ = ['Client_mira']

# %% ../nbs/02_clients.ipynb 3
from fastcore.utils import *
import os
import torch
from collections import OrderedDict
from copy import deepcopy
from peft import *
from .trainers import *
from .utils import get_class
from .data import LLMDataCollator
from transformers import AutoTokenizer
from omegaconf.dictconfig import DictConfig


# %% ../nbs/02_clients.ipynb 20
class Client_mira(BaseClient):
    def __init__(self,
                 data_dict: dict,
                 model: torch.nn.Module,
                 criterion,
                 optimizer: torch.optim.Optimizer,
                 idx: int,
                 gen_data_dict: dict,
                 tokenizer: AutoTokenizer,
                 collat_fn: LLMDataCollator,
                 cfg: DictConfig) -> None:
            
        super().__init__(data_dict, model, criterion, optimizer, idx)
        
        self.train_ds_genr = gen_data_dict['train']
        self.test_ds_genr = gen_data_dict['test']
        self.tokenizer = tokenizer
        self.collat_fn = collat_fn
        self.cfg = cfg 

# %% ../nbs/02_clients.ipynb 22
@patch 
def init_local_train(self: Client_mira, out_dir):

    self.output_dir = out_dir
    self.params_dict_old = deepcopy(
        OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                    "default" in name))
    
    self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                                        "default" in name)
    
    self.model.state_dict = (
        lambda instance, *_, **__: get_peft_model_state_dict(
            instance, self.params_dict_new, "default"
        )
    ).__get__(self.model, type(self.model))

    self.optimizer = get_class('torch.optim', self.cfg.optimizer)(self.model.parameters(), lr= self.cfg.lr)

# %% ../nbs/02_clients.ipynb 23
@patch
def train(self: Client_mira, n_epochs):
    self.model.train()
    trainer = Trainer(self)
    history = trainer.fit(n_epochs)
    return history

# %% ../nbs/02_clients.ipynb 24
@patch
def clear_model(self: Client_mira):
    self.model = None

# %% ../nbs/02_clients.ipynb 25
@patch
def terminate_local_train(self: Client_mira, epoch, local_dataset_len_dict, previously_selected_clients_set):

    local_dataset_len_dict[self.idx] = len(self.train_ds)
    new_adapter_weight = self.model.state_dict()
    single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.idx))
    os.makedirs(single_output_dir, exist_ok=True)
    torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")

    older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
    set_peft_model_state_dict(self.model, older_adapter_weight, "default")
    previously_selected_clients_set = previously_selected_clients_set | set({self.idx})
    last_client_id = self.idx

    return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id
