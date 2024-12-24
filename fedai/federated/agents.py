"""The core abstraction for different FL Clients."""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/02_federated.agents.ipynb.

# %% auto 0
__all__ = ['AgentRole', 'Agent', 'FLAgent', 'PeftAgent', 'AgentMira']

# %% ../../nbs/02_federated.agents.ipynb 3
from fastcore.utils import *
import os
import json
from collections import defaultdict,OrderedDict
from copy import deepcopy
from enum import Enum
import torch
from peft import *
from ..trainers import *
from ..utils import *
from ..data.core import LLMDataCollator
from transformers import AutoTokenizer
from omegaconf.dictconfig import DictConfig


# %% ../../nbs/02_federated.agents.ipynb 4
class AgentRole(Enum):
    SERVER = 1
    CLIENT = 2
    MARL = 3

# %% ../../nbs/02_federated.agents.ipynb 7
class Agent:
    def __init__(self,
                 id,
                 cfg,
                 state= None,
                 role= AgentRole.CLIENT):
        
        self.cfg = cfg # contains all the configurations needed for the agent/trainer.
        self.state = state # A dictionary containing the state of the agent
        self.id = id # each agent has a unique id
        self.role = role # either a client or a server

# %% ../../nbs/02_federated.agents.ipynb 8
@patch
def init_agent(self: Agent):
    # Initialize the state of the agent. In FL Agent, this means making any adjustments to the model/optimizer/state_dict/...etc
    raise NotImplementedError

# %% ../../nbs/02_federated.agents.ipynb 9
@patch
def communicate(self: Agent, msg):
    raise NotImplementedError

# %% ../../nbs/02_federated.agents.ipynb 10
@patch
def update_state(self: Agent):
    raise NotImplementedError

# %% ../../nbs/02_federated.agents.ipynb 11
@patch
def save_state(self: Agent):
    # save the state of the agent to a file on disk (id, model, optimizer, loss_fn).
    raise NotImplementedError

# %% ../../nbs/02_federated.agents.ipynb 12
@patch
def clear_model(self: Agent):
    self.model = None

# %% ../../nbs/02_federated.agents.ipynb 20
class FLAgent(Agent):
    # A Federated Learning Agent that can be used to train a model in a federated learning setting
    def __init__(self,
                 id, # the id of the agent
                 cfg, # the configuration of the agent.
                 state= None, # the state of the agent (model, optimizer, loss_fn), etc.
                 role= AgentRole.CLIENT, # the role of the agent (client or server)
                 block= None): # The data block (local data of the FL Agent).
                 
        super().__init__(id, cfg, state, role)
        if block:
            self.train_ds, self.test_ds = block[0], block[1]
        
        if self.state :
            for key, value in self.state.items():
                setattr(self, key, value)
            self.init_agent()

# %% ../../nbs/02_federated.agents.ipynb 24
@patch
def __str__(self: FLAgent) -> str:
    return f'''FLAgent: {self.__class__.__name__}
    Index : {self.id}
    Model: {self.model.__class__.__name__}
    Criterion: {self.criterion.__class__.__name__}
    Optimizer: {self.optimizer.__class__.__name__}'''


# %% ../../nbs/02_federated.agents.ipynb 25
@patch
def init_agent(self: FLAgent):  # noqa: F811
    self.optimizer = get_class('torch.optim', self.cfg.optimizer.name)(self.model.parameters(),  # noqa: F405
                                                                                lr= self.cfg.lr)

# %% ../../nbs/02_federated.agents.ipynb 26
@patch
def clear_model(self: FLAgent):
    self.model = None if hasattr(self, 'model') else None

# %% ../../nbs/02_federated.agents.ipynb 27
@patch
def save_state(self: FLAgent, state_dict, comm_round):  # noqa: F811
    # save the model to self.cfg.save_dir/comm_round/f"local_output_{id}"/pytorch_model.bin
    
    model_path = os.path.join(self.cfg.save_dir, 
                              str(comm_round),
                              f"local_output_{self.id}")
    
    os.makedirs(model_path, exist_ok=True)
    torch.save(state_dict, 
               os.path.join(model_path, 
                            "pytorch_model.pth"))
    print(f"role is {self.role}")
    if self.role == AgentRole.CLIENT:
        print(f"role is {self.role}")
        save_space(self)


# %% ../../nbs/02_federated.agents.ipynb 30
@patch
def communicate(self: Agent, another_agent: Agent, comm_round):  # noqa: F811
    if self.role == AgentRole.CLIENT:
        self.save_state(self.model.state_dict(), comm_round)

# %% ../../nbs/02_federated.agents.ipynb 31
@patch
def aggregate(self: FLAgent, lst_active_ids, comm_round, len_clients_ds):
    # load the models of the agents in lst_active_ids and `FedAvg` them. At the end, save the aggregated model to the disk.
        
    for i, id in enumerate(lst_active_ids):
        model_path = os.path.join(self.cfg.save_dir, 
                                   str(comm_round),
                                   f"local_output_{id}",
                                   "pytorch_model.pth")
        client_state_dict = torch.load(model_path, map_location='cpu')

        if i == 0:
            client_avg = {
                key: torch.zeros_like(value) 
                for key, value in client_state_dict.items()
            }
        
        weight = len_clients_ds[i] / sum(len_clients_ds)

        for key in client_state_dict.keys():
            client_avg[key].data += weight * client_state_dict[key].data

    for key in client_avg.keys():
        client_avg[key].data /= len(lst_active_ids)

    for id in lst_active_ids:
        model_path = os.path.join(self.cfg.save_dir, 
                                  str(comm_round),
                                  f"local_output_{id}",
                                  "pytorch_model.pth")
        self.save_state(client_avg, comm_round)
    

# %% ../../nbs/02_federated.agents.ipynb 37
class PeftAgent(FLAgent):
    def __init__(self,
                 cfg,
                 block,
                 id,
                 state= None,
                 role= "client",
                 **adapter_params):
        super().__init__(cfg, block, id, state, role)


# %% ../../nbs/02_federated.agents.ipynb 38
@patch
def peftify(self: PeftAgent):
    # extract only the adapter's parameters from the model and store them in a dictionary
    self.params_dict_old = deepcopy(
        OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                    "default" in name))
    
    self.params_dict_new = deepcopy(self.params_dict_old)
    
    self.model.state_dict = (
        lambda instance, *_, **__: get_peft_model_state_dict(  # noqa: F405
            instance, self.params_dict_new, "default"
        )
    ).__get__(self.model, type(self.model))

# %% ../../nbs/02_federated.agents.ipynb 39
@patch 
def init_agent(self: PeftAgent):  # noqa: F811
    self.peftify()
    self.state['optimizer'] = get_class('torch.optim', self.cfg.optimizer.name)(self.model.parameters(),
                                                                                lr= self.cfg.lr)

# %% ../../nbs/02_federated.agents.ipynb 40
@patch
def save_state_(self: PeftAgent, epoch, local_dataset_len_dict, previously_selected_clients_set):  # noqa: F811
    # save the new adapter weights to disk
    self.save_state(epoch)

    local_dataset_len_dict[self.id] = len(self.block)
    older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")  # noqa: F405
    set_peft_model_state_dict(self.model, older_adapter_weight, "default")  # noqa: F405
    previously_selected_clients_set = previously_selected_clients_set | set({self.id})
    last_client_id = self.id

    return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id

# %% ../../nbs/02_federated.agents.ipynb 41
@patch
def strategy(self: PeftAgent):
    # implement the strategy for the agent if it's a server. This is the aggregation strategy.
    pass

# %% ../../nbs/02_federated.agents.ipynb 48
class AgentMira(FLAgent):
    def __init__(self,
                 data_dict: dict,
                 model: torch.nn.Module,
                 criterion,
                 optimizer: torch.optim.Optimizer,
                 id: int,
                 gen_data_dict: dict,
                 tokenizer: AutoTokenizer,
                 collat_fn: LLMDataCollator,
                 cfg: DictConfig) -> None:
            
        super().__init__(data_dict, model, criterion, optimizer, id)
        
        self.train_ds_genr = gen_data_dict['train']
        self.test_ds_genr = gen_data_dict['test']
        self.tokenizer = tokenizer
        self.collat_fn = collat_fn
        self.cfg = cfg 
