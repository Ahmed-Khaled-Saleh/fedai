"""Fill in a module description here"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_servers.ipynb.

# %% auto 0
__all__ = ['BaseServer', 'Server_mira']

# %% ../nbs/01_servers.ipynb 3
from copy import deepcopy
import os
import numpy as np
from collections import defaultdict
import torch
from fastcore.utils import *
from peft import *
from .models import *
from .utils import *
from .clients import Client_mira, BaseClient

# %% ../nbs/01_servers.ipynb 5
class BaseServer:

    def __init__(self, cfg, lst_data_dict, model, holdout_ds, client_class):
        self.cfg = cfg
        self.lst_data_dict = lst_data_dict
        self.model = model
        self.holdout_ds = holdout_ds
        self.client_list = LazyList(self, client_class)  # type: ignore # noqa: F405
        self.latest_model_iter = dict()
        self.__str__ = self.__repr__
       
    def __str__(self) -> str:
        return f'''Server: {self.__class__.__name__}'''

# %% ../nbs/01_servers.ipynb 7
@patch
def send(self: BaseServer, client: BaseClient):  # noqa: F811
    
    if client.idx in self.latest_model_iter:
        comm_round = self.latest_model_iter[client.idx]
        model_path = os.path.join(self.cfg.output_dir, str(comm_round), 
                                  "local_output_{}".format(client.idx),
                                  "pytorch_model.pth")
    else:
        model_path = ''

    with torch.no_grad():
        client.model = deepcopy(self.model)
    
    if os.path.exists(model_path):
        if isinstance(client.model, torch.nn.Module):
            client.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        elif isinstance(client.model, PeftModel): # noqa: F405
            set_peft_model_state_dict(client.model,  # noqa: F405
                                  torch.load(model_path, map_location='cpu'),
                                  "default")
    return client.model

# %% ../nbs/01_servers.ipynb 9
@patch
def client_selection(self: BaseServer):
    client_indices_rounds = []
    for _ in range(self.cfg.rounds):
        client_indices_rounds.append(np.random.choice(a= np.arange(self.cfg.num_clients), 
                                                      size=int(self.cfg.num_clients * self.cfg.m), 
                                                      replace=False))
        
    return client_indices_rounds

# %% ../nbs/01_servers.ipynb 11
@patch
def get_selected_client(self: BaseServer,
                        client_indices: list) : # a list of current round's selected clients
    
    for idx in client_indices:
        yield self.client_list[idx]  # Lazily access the client and return a generator


# %% ../nbs/01_servers.ipynb 14
class Server_mira(BaseServer):
    def __init__(self, cfg, lst_data_dict, model, holdout_ds,client_class, **kwargs):
        super().__init__(cfg, lst_data_dict, model, holdout_ds, client_class)
        
        self.model = get_model(self.cfg)
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.model.resize_token_embeddings(len(self.tokenizer))

# %% ../nbs/01_servers.ipynb 16
@patch
def init_sim_matrix(self: Server_mira):
    N = self.cfg.num_clients
    b = np.random.uniform(0,1,size=(N,N))
    b_symm = (b + b.T)/2
    b_symm[b_symm < 0.25] = 0
    self.alk_connection = b_symm


# %% ../nbs/01_servers.ipynb 23
@patch
def aggregate(self: Server_mira, selected_clients_indices, comm_round):
    global_lr = float(self.cfg.lr) * float(self.cfg.local_step)

    for i, client_id in enumerate(selected_clients_indices):
        client_path = os.path.join(self.cfg.output_dir, str(comm_round), f"local_output_{client_id}", "pytorch_model.pth")
        client_state_dict = torch.load(client_path, map_location=self.device)

        client_diff = defaultdict(lambda: torch.tensor(0.0).to(self.device))

        for key in client_state_dict.keys():
            client_diff[key] = torch.zeros_like(client_state_dict[key]).to(self.device)

        for j, other_client_id in enumerate(selected_clients_indices):
            if i != j:
                other_client_path = os.path.join(self.cfg.output_dir, str(comm_round), f"local_output_{other_client_id}", "pytorch_model.pth")
                other_client_state_dict = torch.load(other_client_path, map_location=self.device)

                weight = self.alk_connection[int(client_id)][int(other_client_id)]
                for key in client_state_dict.keys():
                    client_diff[key].data += weight * (client_state_dict[key].data.clone() - other_client_state_dict[key].data.clone())

        for key in client_state_dict:
            client_state_dict[key].data -=  global_lr * self.cfg.lambda_ * client_diff[key].data

        self.update(client_state_dict, comm_round, client_id)


# %% ../nbs/01_servers.ipynb 24
@patch
def update(self: Server_mira, client_state_dict: dict, comm_round: int, client_id: int) -> None:
    save_dir = os.path.join(self.cfg.output_dir, str(comm_round + 1), f"local_output_{client_id}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "pytorch_model.pth")
    torch.save(client_state_dict, save_path)
    set_peft_model_state_dict(self.model, client_state_dict, "default")  # noqa: F405
    self.model.save_pretrained(save_dir)

