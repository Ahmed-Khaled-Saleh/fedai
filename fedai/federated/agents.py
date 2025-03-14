"""The core abstraction for different FL Agents/Clients."""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/02_federated.agents.ipynb.

# %% auto 0
__all__ = ['AgentRole', 'Agent', 'FLAgent', 'Fedu', 'DMTL', 'get_coalitions', 'get_shapley_vals', 'repr_alignment', 'PeftAgent',
           'FedSophiaAgent', 'PadgAgent', 'AgentMira']

# %% ../../nbs/02_federated.agents.ipynb 3
from fastcore.utils import *
from fastcore.all import *
import os
import json
from collections import defaultdict,OrderedDict
from copy import deepcopy
from enum import Enum
import torch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from peft import *
from community import community_louvain
from ..utils import *
from ..client_selector import *
from ..optimizers import *
from ..data.core import LLMDataCollator
from tqdm import tqdm
import numpy as np
from loguru import logger
from ..utils import *
from ..metrics import *
from transformers import AutoTokenizer
from omegaconf.dictconfig import DictConfig
import numpy as np
np.random.seed(42)
torch.manual_seed(42)

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
        
        self.id = id # each agent has a unique id
        self.cfg = cfg # contains all the configurations needed for the agent/trainer.
        self.state = state # A dictionary containing the state of the agent
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

# %% ../../nbs/02_federated.agents.ipynb 19
class FLAgent(Agent):
    # A Federated Learning Agent implementing `FedAVG`.
    def __init__(self,
                 id, # the id of the agent
                 cfg, # the configuration of the agent.
                 state= None, # the state of the agent (model, optimizer, loss_fn), etc.
                 role= AgentRole.CLIENT, # the role of the agent (client or server)
                 block= None # The data block (local data of the FL Agent).
                 ):  
                 
        super().__init__(id, cfg, state, role)

        if self.role == AgentRole.CLIENT:
            self.train_ds, self.test_ds = block[0], block[1]
            
            for key, value in self.state.items():
                setattr(self, key, value)

            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
            self.train_loader = prepare_dl(self.cfg, self.train_ds)  # noqa: F405
            self.test_loader = prepare_dl(self.cfg, self.test_ds) # noqa: F405

            self.training_metrics = Metrics(list(self.cfg.training_metrics))  # noqa: F405
            self.test_metrics = Metrics(list(self.cfg.test_metrics))  # noqa: F405

            self.data_key, self.label_key = 'x', 'y'

# %% ../../nbs/02_federated.agents.ipynb 20
@patch
def server_init(self: FLAgent, client_fn, client_selector, client_cls, loss_fn, writer):
    self.client_fn = client_fn
    self.client_selector = client_selector
    self.client_cls = client_cls
    self.loss_fn = loss_fn
    self.writer = writer
    self.latest_round = {}

# %% ../../nbs/02_federated.agents.ipynb 23
@patch
def runFL(self: FLAgent):
    res =  []
    all_ids = self.client_selector.select()
    
    for t in range(1, self.cfg.n_rounds):
        lst_active_ids = all_ids[t]
        len_clients_ds = []
        round_res = []

        for id in lst_active_ids:
            client = self.client_fn(self.client_cls, self.cfg, id, self.latest_round, t, self.loss_fn)
            len_clients_ds.append(len(client.train_ds))
            
            self.communicate(client) 

            client_history = client.fit() 
            round_res.append(client_history)
            res.append(round_res)

            client.communicate(self) 
            self.latest_round[id] = t 

        self.aggregate(lst_active_ids, t, len_clients_ds) 
        self.writer.write(round_res, t) 
        
    self.writer.save(res)
    self.writer.finish()

    return res

# %% ../../nbs/02_federated.agents.ipynb 25
@patch
def __str__(self: FLAgent) -> str:
    return f'''{self.__class__.__name__}: {self.__class__.__name__}
    Index : {self.id}
    Model: {self.model.__class__.__name__}
    Criterion: {self.criterion.__class__.__name__}
    Optimizer: {self.optimizer.__class__.__name__}'''


# %% ../../nbs/02_federated.agents.ipynb 26
@patch
def clear_model(self: FLAgent):
    self.model = None if hasattr(self, 'model') else None

# %% ../../nbs/02_federated.agents.ipynb 27
@patch
def get_batch(self: FLAgent, batch):
    return {k: v.to(self.device) for k, v in batch.items()}

# %% ../../nbs/02_federated.agents.ipynb 28
@patch
def _forward(self: FLAgent, batch):
    X, y = batch['x'], batch['y']
    outputs = self.model(X)
    loss = self.criterion(outputs, y)
    return loss, outputs

# %% ../../nbs/02_federated.agents.ipynb 29
@patch
def _closure(self: FLAgent, batch: dict) -> tuple:
    try:
        loss, logits = self._forward(batch)
        probs =  torch.nn.functional.softmax(logits, dim= -1)
        y_pred = probs.argmax(dim= -1)
        y_true = batch[self.label_key]

        if self.cfg.training_metrics:
            if hasattr(self, "tokenizer"):
                metrcis = self.training_metrics.compute(y_pred= y_pred, y_true= y_true, tokenizer= self.tokenizer)
            else:
                metrcis = self.training_metrics.compute(y_pred= y_pred, y_true= y_true)
                
        else:
            metrcis = {k: 0 for k in self.cfg.training_metrics}
            
    except Exception as e:
        metrcis = {k: 0 for k in self.cfg.training_metrics}
        return torch.tensor(float(0), device=self.device), metrcis
        
    return loss, metrcis

# %% ../../nbs/02_federated.agents.ipynb 30
@patch
def _run_batch(self: FLAgent, batch: dict) -> tuple:
    self.model.zero_grad(set_to_none=True)
    loss, metrics = self._closure(batch)

    if loss.item() == 0:
        return loss, metrics
    
    loss.backward()
    if self.cfg.model.grad_norm_clip:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.model.grad_norm_clip)

    self.optimizer.step()

    return loss, metrics

# %% ../../nbs/02_federated.agents.ipynb 31
@patch
def _run_epoch(self: FLAgent):
    total_loss = 0
    lst_metrics= [] 
    num_trained = 0
    progress_bar = tqdm(range(len(self.train_loader)))

    self.model.train()
    for i, batch in enumerate(self.train_loader):
        
        batch = self.get_batch(batch)
        loss, train_metrics = self._run_batch(batch)
        lst_metrics.append(train_metrics)
        
        if num_trained == 0:
            num_trained = 1e-10

        if loss.item() != 0:
            total_loss += loss.item()
            num_trained += len(batch[self.data_key])
            
    epoch_metrics = {k: sum([m[k] for m in lst_metrics]) / len(lst_metrics) for k in self.cfg.training_metrics}

    return total_loss / num_trained, epoch_metrics

# %% ../../nbs/02_federated.agents.ipynb 32
@patch
def fit(self: FLAgent) -> dict:
    
    self.model = self.model.to(self.device)
    test_loss, test_metrics = self.test()
    train_loss = []
    train_metrics = []
    for _ in range(self.cfg.local_epochs):
        
        avg_train_loss, metrics_train = self._run_epoch()
        train_loss.append(avg_train_loss)
        train_metrics.append(metrics_train)

    # average the metrics across all local rounds to get local train metrics (e.g, train accuracy)
    train_metrics = {k: sum([m[k] for m in train_metrics]) / len(train_metrics) for k in train_metrics[0].keys()}

    train_metrics = {f'train_{k}': v for k, v in train_metrics.items()}
    test_metrics = {f'test_{k}': v for k, v in test_metrics.items()}

    history =  {
        'train_loss': np.mean(train_loss),
        'test_loss': test_loss,
    }

    history.update(train_metrics)
    history.update(test_metrics)
    
    return history
    


# %% ../../nbs/02_federated.agents.ipynb 33
@patch
def test(self: FLAgent) -> dict:
    total_loss = 0
    lst_metrics = []

    self.model = self.model.to(self.device)
    self.model.eval()
    num_eval = 0
    
    with torch.no_grad():
        for i, batch in enumerate(self.test_loader):
            
            batch = self.get_batch(batch)

            if num_eval == 0:
                num_eval = 1e-10

            test_loss, test_metrics = self._closure(batch)                 

            if (not torch.isnan(test_loss)) and (self.cfg.model.grad_norm_clip <= 0 or test_loss != 0.0):
                total_loss += test_loss.item()  
                num_eval += len(batch[self.data_key])
                lst_metrics.append(test_metrics)           
            
    total_test_metrics = {k: sum([m[k] for m in lst_metrics]) / len(lst_metrics) for k in lst_metrics[0].keys()}

    return total_loss / num_eval, total_test_metrics

# %% ../../nbs/02_federated.agents.ipynb 34
@patch
def save_state(self: FLAgent, state_dict):  # noqa: F811
    # save the model to self.cfg.save_dir/comm_round/f"local_output_{id}"/state.pth
    
    state_path = os.path.join(self.cfg.save_dir, 
                              str(self.t),
                              f"local_output_{self.id}",
                              "state.pth")
    if not os.path.exists(os.path.dirname(state_path)):
        os.makedirs(os.path.dirname(state_path))

    state_dict['model'] = self.model.state_dict()
    state_dict['optimizer'] = self.optimizer.state_dict()

    torch.save(state_dict, state_path)

    if self.role == AgentRole.CLIENT:
        save_space(self)


# %% ../../nbs/02_federated.agents.ipynb 37
@patch
def communicate(self: Agent, another_agent: Agent):  # noqa: F811
    if self.role == AgentRole.CLIENT:
        self.save_state(self.state)

# %% ../../nbs/02_federated.agents.ipynb 39
@patch
def aggregate(self: FLAgent, lst_active_ids, comm_round, len_clients_ds):
        
    m_t = sum(len_clients_ds[id] for id in lst_active_ids)

    for i, id in enumerate(lst_active_ids):
        state_path = os.path.join(self.cfg.save_dir, 
                                   str(comm_round),
                                   f"local_output_{id}",
                                   "state.pth")
        
        state = torch.load(state_path, weights_only=False)
        client_state_dict = state['model']

        if i == 0:
            global_model = {
                key: torch.zeros_like(value) 
                for key, value in client_state_dict.items()
            }

        n_k = len_clients_ds[id]
        weight =  n_k / m_t 

        with torch.no_grad():
            for key in client_state_dict.keys():
                global_model[key].add_(weight * client_state_dict[key])


    server_state = {
        'model': global_model,
    }

    server_state_path = os.path.join(self.cfg.save_dir, 
                                  str(comm_round),
                                  "global_model",
                                  "state.pth")
    os.makedirs(os.path.dirname(server_state_path), exist_ok=True)

    torch.save(server_state, server_state_path)

    

# %% ../../nbs/02_federated.agents.ipynb 42
class Fedu(FLAgent):
    def __init__(self, 
                 id,
                 cfg,
                 state= None,
                 role= AgentRole.CLIENT,
                 block= None):
        
        super().__init__(id, cfg, state, role, block)

        self.alk_connection = np.random.rand(self.cfg.num_clients, self.cfg.num_clients)
        self.alk_connection = self.alk_connection / self.alk_connection.sum(axis=1)[:, None]


# %% ../../nbs/02_federated.agents.ipynb 44
@patch
def aggregate(self: Fedu, lst_active_ids, comm_round, len_clients_ds):

    global_lr = float(self.cfg.lr) * float(self.cfg.local_epochs)
    reg_param = self.cfg.lambda_
    for i, id in enumerate(lst_active_ids):
        state_path = os.path.join(self.cfg.save_dir, 
                                   str(comm_round),
                                   f"local_output_{id}",
                                   "state.pth")
        
        state = torch.load(state_path, weights_only= False)
        client_state_dict = state['model']

        client_diff = {
            key: torch.zeros_like(value) 
            for key, value in client_state_dict.items()
        }
        
        for j, other_id in enumerate(lst_active_ids):
            if i == j:
                continue
            other_state_path = os.path.join(self.cfg.save_dir,
                                            str(comm_round),
                                            f"local_output_{other_id}",
                                            "state.pth")
            
            other_state = torch.load(other_state_path, weights_only= False)
            other_state_dict = other_state['model']

            weight = self.alk_connection[int(id)][int(other_id)]
            for key in client_state_dict.keys():
                client_diff[key].data += weight * (client_state_dict[key].data.clone() - other_state_dict[key].data.clone())

        for key in client_state_dict:
            client_state_dict[key].data -=  global_lr * reg_param * client_diff[key].data

        clinet_state = {
            'model': client_state_dict,
        }

        agg_client_state_path = os.path.join(self.cfg.save_dir, 
                                    str(comm_round),
                                    f"aggregated_model_{id}",
                                    "state.pth")
        if not os.path.exists(os.path.dirname(agg_client_state_path)):
            os.makedirs(os.path.dirname(agg_client_state_path))

        torch.save(clinet_state, agg_client_state_path)

# %% ../../nbs/02_federated.agents.ipynb 47
class DMTL(FLAgent):
    def __init__(self, 
                 id,
                 cfg,
                 state= None,
                 role= AgentRole.CLIENT,
                 block= None):
        
        super().__init__(id, cfg, state, role, block)

# %% ../../nbs/02_federated.agents.ipynb 49
@patch
def head_similarity(self: DMTL, model1, model2):
    pass

# %% ../../nbs/02_federated.agents.ipynb 50
@patch
def repr_similarity(self: DMTL, model1, model2):
    pass

# %% ../../nbs/02_federated.agents.ipynb 53
def get_coalitions(self: DMTL):
    return community_louvain.best_partition(self.G)


# %% ../../nbs/02_federated.agents.ipynb 55
def get_shapley_vals(self: DMTL):
    pass

# %% ../../nbs/02_federated.agents.ipynb 56
@patch
def aggregate(self: Fedu, lst_active_ids, comm_round, len_clients_ds):

    global_lr = float(self.cfg.lr) * float(self.cfg.local_epochs)
    reg_param = self.cfg.lambda_
    self.G = self.build_graph(self.cfg)
    partitions = self.get_coalitions()
    # shapely_values = self.get_shapley_vals()

    for k, v in partitions.items():
        for i, id in enumerate(v):
            state_path = os.path.join(self.cfg.save_dir, 
                                    str(comm_round),
                                    f"local_output_{id}",
                                    "state.pth")
            state = torch.load(state_path, weights_only= False)
            client_head = state['model']['head']
            client_repr = state['model']['repr']

            if i == 0:
                client_avg = {
                    key: torch.zeros_like(value) 
                    for key, value in client_head.items()
                }

            for j, other_id in enumerate(v):
                if i == j:
                    continue
                other_state_path = os.path.join(self.cfg.save_dir,
                                                str(comm_round),
                                                f"local_output_{other_id}",
                                                "state.pth")
                other_state = torch.load(other_state_path, weights_only= False)
                other_head = other_state['model']['head']
                other_repr = other_state['model']['repr']



        client_diff = {
            key: torch.zeros_like(value) 
            for key, value in client_head.items()
        }
        
        for j, other_id in enumerate(lst_active_ids):
            if i == j:
                continue
            other_state_path = os.path.join(self.cfg.save_dir,
                                            str(comm_round),
                                            f"local_output_{other_id}",
                                            "state.pth")
            
            other_state = torch.load(other_state_path, weights_only= False)
            other_state_dict = other_state['model']

            weight = self.alk_connection[int(id)][int(other_id)] #FIXME
            for key in client_state_dict.keys():
                client_diff[key].data += weight * (client_state_dict[key].data.clone() - other_state_dict[key].data.clone())

    for key in client_state_dict:
        client_state_dict[key].data -=  global_lr * reg_param * client_diff[key].data

    clinet_state = {
        'model': client_state_dict,
    }

    agg_client_state_path = os.path.join(self.cfg.save_dir, 
                                  str(comm_round),
                                  f"aggregated_model_{id}",
                                  "state.pth")
    if not os.path.exists(os.path.dirname(agg_client_state_path)):
        os.makedirs(os.path.dirname(agg_client_state_path))

    torch.save(clinet_state, agg_client_state_path)

# %% ../../nbs/02_federated.agents.ipynb 58
def repr_alignment(self: DMTL):
    pass

# %% ../../nbs/02_federated.agents.ipynb 60
class PeftAgent(FLAgent):
    def __init__(self,
                 cfg,
                 block,
                 id,
                 state= None,
                 role= "client",
                 **adapter_settings):
        super().__init__(cfg, block, id, state, role)


# %% ../../nbs/02_federated.agents.ipynb 61
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

# %% ../../nbs/02_federated.agents.ipynb 62
@patch 
def init_agent(self: PeftAgent):  # noqa: F811
    self.peftify()


# %% ../../nbs/02_federated.agents.ipynb 63
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

# %% ../../nbs/02_federated.agents.ipynb 65
class FedSophiaAgent(FLAgent):
    def __init__(self,
                 id, # the id of the agent
                 cfg, # the configuration of the agent.
                 state= None, # the state of the agent (model, optimizer, loss_fn), etc.
                 role= AgentRole.CLIENT, # the role of the agent (client or server)
                 block= None):
        super().__init__(id, cfg, state, role, block)


# %% ../../nbs/02_federated.agents.ipynb 66
@patch
def train(self: FedSophiaAgent):
    trainer = self.trainer(self) 
    client_history = trainer.fit() 
    return client_history

# %% ../../nbs/02_federated.agents.ipynb 68
class PadgAgent(FLAgent):
    def __init__(self,
                 id, # the id of the agent
                 cfg, # the configuration of the agent.
                 state= None, # the state of the agent (model, optimizer, loss_fn), etc.
                 role= AgentRole.CLIENT, # the role of the agent (client or server)
                 block= None):
        super().__init__(id, cfg, state, role, block)

        if role == AgentRole.SERVER:
            self.connections = torch.from_numpy(generate_graph(self.cfg.num_clients))  # noqa: F405


# %% ../../nbs/02_federated.agents.ipynb 69
@patch
def apply_constraints(self: PadgAgent, 
                      graph, # (np.ndarray): The input matrix.
                      symmetrize=True, # (bool): If True, makes the matrix symmetric.
                      normalize=True, # (bool): If True, normalizes the matrix symmetrically.
                      threshold= 0, # (float or None): If provided, sets values below this threshold to 0.
                      diag_fill= 0): # (float or None): If provided, fills the diagonal with this value.
    

    # Symmetrize the matrix
    if symmetrize:
        graph = (graph + graph.T) / 2

    # Apply threshold to ensure non-negativity
    if threshold is not None:
        graph = torch.where(graph > threshold, graph, 0)

    # Normalize the matrix symmetrically
    if normalize:
        row_sums = graph.sum(axis=1, keepdims=True)
        col_sums = graph.sum(axis=0, keepdims=True)
        norm_factor = torch.sqrt(row_sums @ col_sums)  # Symmetric normalization factor
        graph = torch.divide(graph, norm_factor, where=norm_factor != 0)

    # Fill the diagonal
    if diag_fill is not None:
        torch.fill_diagonal(graph, diag_fill)

    return graph


# %% ../../nbs/02_federated.agents.ipynb 73
@patch
def compute_probs(self: PadgAgent,
                  batch_size=32, # batch_size (int): Batch size for evaluation.
                  return_log_probs=True): # return_log_probs (bool): If True, return log-probabilities; otherwise, return probabilities.
    
    # Computes probabilities or log-probabilities across the entire dataset for a given model.
    # Ensure model is in evaluation mode
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.model.to(device)
    self.model.eval()
    
    # Create DataLoader for the dataset
    dataloader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=False)
    
    all_probs = []  # To store probabilities or log-probabilities for all batches
    
    with torch.no_grad():  # Disable gradient computation
        for batch in dataloader:  # Assuming dataset returns (inputs, labels)
            inputs = batch['x'].to(device)  # Move to model's device
            
            logits = self.model(inputs)
            
            if return_log_probs:
                # Convert logits to log-probabilities
                batch_log_probs = F.log_softmax(logits, dim=-1)
                all_probs.append(batch_log_probs)
            else:
                # Convert logits to probabilities
                batch_probs = F.softmax(logits, dim=-1)
                all_probs.append(batch_probs)
    
    self.model.to('cpu')
    # Concatenate all batch probabilities/log-probabilities
    return torch.cat(all_probs, dim=0)


# %% ../../nbs/02_federated.agents.ipynb 75
@patch
def aggregate(self: PadgAgent, lst_active_ids, comm_round, len_clients_ds, one_model= False):
    
    visited = []
    for i, id in enumerate(lst_active_ids):

        neighbour_ids = torch.where(self.connections[id] != float(0))[0]

        model_path = os.path.join(self.cfg.save_dir, 
                                   str(comm_round),
                                   f"local_output_{id}",
                                   "pytorch_model.pth")
        client_state_dict = torch.load(model_path, map_location='cpu', weights_only= False)
        self.model.load_state_dict(client_state_dict)
        
        neighbours_sum = {
            key: torch.zeros_like(value) 
            for key, value in client_state_dict.items()
        }
            
        probs_1 = self.compute_probs(batch_size=32, return_log_probs=True)
        
        for other_id in neighbour_ids:

            if (other_id, id) in visited:
                continue
            other_model_path = os.path.join(self.cfg.save_dir, 
                                    str(comm_round),
                                    f"local_output_{other_id}",
                                    "pytorch_model.pth")
            
            other_client_state_dict = torch.load(other_model_path, map_location='cpu', weights_only= False)
            self.model.load_state_dict(other_client_state_dict)
            
            probs_2 = self.compute_probs(batch_size=32, return_log_probs=False)

            kl_div = F.kl_div(probs_1, probs_2, reduction= 'batchmean').to('cpu')
            self.connections[id][other_id] -= self.cfg.server_lr * self.cfg.lambda_ * kl_div

            # apply constraints to the KL divergence
            self.connections[id][other_id] = self.apply_constraints(self.connections[id][other_id])
            self.connections[id][other_id] = self.connections[other_id][id]

            visited.append((id, other_id))
            visited.append((other_id, id))
            
        for other_id in neighbour_ids:
            other_model_path = os.path.join(self.cfg.save_dir, 
                                    str(comm_round),
                                    f"local_output_{other_id}",
                                    "pytorch_model.pth")
            other_client_state_dict = torch.load(other_model_path, map_location='cpu', weights_only= False)

            weight = self.connections[id][other_id]
            for key in other_client_state_dict.keys():
                neighbours_sum[key].data += weight * other_client_state_dict[key].data

        # for key in neighbours_sum.keys():
            # neighbours_sum[key].data /= len(neighbour_ids)

        for key in client_state_dict.keys():
            client_state_dict[key].data = self.cfg.beta * client_state_dict[key].data + (1 - self.cfg.beta) * neighbours_sum[key].data

    
        # save the updated model to the disk
        self.save_state(client_state_dict, comm_round + 1, id)
        

# %% ../../nbs/02_federated.agents.ipynb 80
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
