"""Implementing the training loop for CPU/GPU training in pytorch."""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_trainers.ipynb.ipynb.

# %% auto 0
__all__ = ['Trainer', 'LLMTrainer', 'FedSophiaTrainer']

# %% ../nbs/03_trainers.ipynb.ipynb 3
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from fastcore.all import *
from fastcore.utils import *
from loguru import logger
from .utils import *
from .metrics import *


# %% ../nbs/03_trainers.ipynb.ipynb 8
class Trainer:
    def __init__(self, client):
        self.client = client
        self.cfg = client.cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.client.train_loader = prepare_dl(self.cfg, self.client.train_ds)  # noqa: F405
        self.client.test_loader = prepare_dl(self.cfg, self.client.test_ds) # noqa: F405

        self.training_metrics = Metrics(list(self.cfg.training_metrics))  # noqa: F405
        self.test_metrics = Metrics(list(self.cfg.test_metrics))  # noqa: F405

        self.data_key, self.label_key = 'x', 'y'

# %% ../nbs/03_trainers.ipynb.ipynb 10
@patch
def get_batch(self: Trainer, batch):
    return {k: v.to(self.device) for k, v in batch.items()}

# %% ../nbs/03_trainers.ipynb.ipynb 12
@patch
def _forward(self: Trainer, batch):
    X, y = batch['x'], batch['y']
    outputs = self.client.model(X)
    loss = self.client.criterion(outputs, y)
    return loss, outputs

# %% ../nbs/03_trainers.ipynb.ipynb 13
@patch
def _closure(self: Trainer, batch: dict) -> tuple:
    try:
        loss, logits = self._forward(batch)
        probs =  torch.nn.functional.softmax(logits, dim= -1)
        y_pred = probs.argmax(dim= -1)#.view(-1)
        y_true = batch[self.label_key]#.view(-1)

        if self.cfg.training_metrics:

            if hasattr(self.client, "tokenizer"):
                metrcis = self.training_metrics.compute(y_pred= y_pred, y_true= y_true, tokenizer= self.client.tokenizer)
            else:
                metrcis = self.training_metrics.compute(y_pred= y_pred, y_true= y_true)
                
        else:
            metrcis = {k: 0 for k in self.training_metrics}
            
    except Exception as e:
        # print(f"Error in loss calculation: {e}")
        metrcis = {k: 0 for k in self.training_metrics}
        return torch.tensor(float(0), device=self.device), metrcis
        
    return loss, metrcis

# %% ../nbs/03_trainers.ipynb.ipynb 14
@patch
def _run_batch(self: Trainer, batch: dict) -> tuple:
    loss, metrics = self._closure(batch)
    self.client.model.zero_grad(set_to_none=True)

    if loss.item() == 0:
        return loss, metrics
    
    loss.backward()
    if self.cfg.model.grad_norm_clip:
        torch.nn.utils.clip_grad_norm_(self.client.model.parameters(), self.cfg.model.grad_norm_clip)

    self.client.optimizer.step()

    return loss, metrics

# %% ../nbs/03_trainers.ipynb.ipynb 15
@patch
def _run_epoch(self: Trainer):
    total_loss = 0
    lst_metrics= [] 
    num_trained = 0
    progress_bar = tqdm(range(len(self.client.train_loader)))

    self.client.model.train()
    for i, batch in enumerate(self.client.train_loader):
            
        batch = self.get_batch(batch)
        loss, metrics = self._run_batch(batch)

        if num_trained == 0:
            num_trained = 1e-10

        # print(f'Batch loss is {loss}')
        # progress_bar.update(1)
        # progress_bar.set_description(f'client {self.client.id} total_loss at step {i}, loss: {total_loss / num_trained if num_trained != 0 else 0.0}')
        
        if loss.item() != 0:
            total_loss += loss.item()
            num_trained += len(batch[self.data_key])
            lst_metrics.append(metrics)

    # average the metrics of the epoch (across batches)
    epoch_metrics = {k: sum([m[k] for m in lst_metrics]) / len(lst_metrics) for k in lst_metrics[0].keys()}

    return total_loss / num_trained, epoch_metrics

# %% ../nbs/03_trainers.ipynb.ipynb 16
@patch
def train(self: Trainer) -> dict:
    init_test_loss, metrics_test = self.test()
    test_loss = [init_test_loss]
    test_metrics = [metrics_test]
    
    train_loss = []
    train_metrics = []
    for _ in range(self.cfg.local_epochs):
        
        self.client.model = self.client.model.to(self.device)

        avg_train_loss, metrics_train = self._run_epoch()
        train_loss.append(avg_train_loss)
        train_metrics.append(metrics_train)
        
        avg_test_loss, metrics_test = self.test()
        test_loss.append(avg_test_loss)   
        test_metrics.append(metrics_test)

    # average the metrics across all local rounds
    train_metrics = {k: sum([m[k] for m in train_metrics]) / len(train_metrics) for k in train_metrics[0].keys()}
    test_metrics = {k: sum([m[k] for m in test_metrics]) / len(test_metrics) for k in test_metrics[0].keys()}

    # add train_ to any key in train_metrics
    train_metrics = {f'train_{k}': v for k, v in train_metrics.items()}
    test_metrics = {f'test_{k}': v for k, v in test_metrics.items()}

    all_metrics =  {
        'train_loss': np.mean(train_loss),
        'test_loss': np.mean(test_loss),
    }

    # add the train and test metrics to the losses
    all_metrics.update(train_metrics)
    all_metrics.update(test_metrics)
    
    return all_metrics
    


# %% ../nbs/03_trainers.ipynb.ipynb 18
@patch
def test(self: Trainer) -> dict:
    total_loss = 0
    lst_metrics = []
    # print("****************************************")
    # print(f'Inside the test () function of client {self.client.id}')
    
    self.client.model = self.client.model.to(self.device)
    self.client.model.eval()
    num_eval = 0
    
    with torch.no_grad():
        for i, batch in enumerate(self.client.test_loader):
            
            batch = self.get_batch(batch)

            if num_eval == 0:
                num_eval = 1e-10

            loss, metrics = self._closure(batch)                 

            # print(f"Client {self.client.id}'s Batch loss inside eval() : {loss}")

            if (not torch.isnan(loss)) and (self.cfg.model.grad_norm_clip <= 0 or loss != 0.0):
                total_loss += loss.item()  
                num_eval += len(batch[self.data_key])
                lst_metrics.append(metrics)           
            
        # print(f'Client {self.client.id} Eval loss is : {total_loss / num_eval}')
        # print("****************************************")
    epoch_metrics = {k: sum([m[k] for m in lst_metrics]) / len(lst_metrics) for k in lst_metrics[0].keys()}

    return total_loss / num_eval, epoch_metrics

# %% ../nbs/03_trainers.ipynb.ipynb 21
class LLMTrainer(Trainer):
    def __init__(
        self,
        client
    ) -> None:        
        super().__init__(client)
        
        self.client.train_iterator = iter(self.client.train_loader)
        self.client.model.generation_config.pad_token_id = self.client.tokenizer.pad_token_id
        self.data_key, self.label_key = 'input_ids', 'labels'

# %% ../nbs/03_trainers.ipynb.ipynb 22
@patch
def get_batch(self: LLMTrainer, batch):  # noqa: F811
    return {
                'input_ids': batch['input_ids'].to(self.device),
                'labels': batch['labels'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device) 
            }
    

# %% ../nbs/03_trainers.ipynb.ipynb 23
@patch
def _forward(self: LLMTrainer, batch):
    outputs = self.client.model(**batch)
    loss = self.client.criterion(outputs)
    return loss, outputs.logits

# %% ../nbs/03_trainers.ipynb.ipynb 26
@patch
def test_generate(self: LLMTrainer) -> float:
    # print("****************************************")
    # print(f'Inside the test_generate () function of client {self.client.id}')

    lst_metrics = []
    self.client.model = self.client.model.to(self.device)
    self.client.model.eval()
    
    progress_bar_eval = tqdm(range(len(self.client.test_loader_genr)))

    with torch.no_grad():
        for batch in self.client.test_loader_genr:

            batch = self.get_batch(batch)

            output_ids = self.client.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_new_tokens=self.cfg.max_new_tokens,
                num_beams=self.cfg.num_beams,
            )

            generated_ids = output_ids[:, len(batch['input_ids'][0]):] 
            metrics = self.test_metrics.compute(y_pred= generated_ids,
                                                y_true= batch['labels'],
                                                tokenizer= self.client.tokenizer)
    
            lst_metrics.append(metrics)

            # print(f"Client {self.client.id}'s Batch test metrics are : {metrics}")
            progress_bar_eval.update(1)

    # print("****************************************")
    epoch_metrics = {k: sum([m[k] for m in lst_metrics]) / len(lst_metrics) for k in lst_metrics[0].keys()}

    return epoch_metrics


# %% ../nbs/03_trainers.ipynb.ipynb 29
class FedSophiaTrainer(Trainer):
    def __init__(
        self,
        client
    ) -> None:        
        super().__init__(client)
        self.t = self.client.t
        self.client.train_iterator = iter(self.client.train_loader)


# %% ../nbs/03_trainers.ipynb.ipynb 30
@patch
def get_next_train_batch(self: FedSophiaTrainer):
    try:
        batch = next(self.train_iterator)
        batch = self.get_batch(batch)
    except StopIteration:
        self.train_iterator = iter(self.trainloader)
        batch = next(self.train_iterator)
        batch = self.get_batch(batch)
    return batch

# %% ../nbs/03_trainers.ipynb.ipynb 31
@patch
def estimate_hessian(self: FedSophiaTrainer):
    "Gauss Newton Barlette hessian estimator"
    batch = self.get_next_train_batch()
    X = batch['x']
    grad_clip = 1
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(self.client.model.parameters(), grad_clip)
                
    logits = self.client.model(X) 
    samp_dist = torch.distributions.Categorical(logits=logits)
    y_sample = samp_dist.sample()
    loss_sampled = F.cross_entropy(logits.view(-1, logits.size(-1)), y_sample.view(-1), ignore_index=-1)
    loss_sampled.backward()
    self.client.optimizer.update_hessian()
    self.cliet.optimizer.zero_grad(set_to_none=True)
    self.client.model.zero_grad()

# %% ../nbs/03_trainers.ipynb.ipynb 32
@patch
def _run_batch(self: FedSophiaTrainer, batch: dict) -> tuple:
    loss, metrics = self._closure(batch)
    self.client.optimizer.zero_grad(set_to_none=True)

    if loss.item() == 0:
        return loss, metrics
    
    loss.backward()
    if self.cfg.model.grad_norm_clip:
        torch.nn.utils.clip_grad_norm_(self.client.model.parameters(), self.cfg.model.grad_norm_clip)

    self.client.optimizer.step()

    return loss, metrics

# %% ../nbs/03_trainers.ipynb.ipynb 33
@patch
def _run_epoch(self: FedSophiaTrainer):
    total_loss = 0
    lst_metrics= [] 
    num_trained = 0
    progress_bar = tqdm(range(len(self.client.train_loader)))

    self.client.model.train()
    batch = self.get_next_train_batch()
    loss, metrics = self._run_batch(batch)

    if self.t % self.cfg.tau == 0:
        self.estimate_hessian()

    if num_trained == 0:
        num_trained = 1e-10

    if loss.item() != 0:
        total_loss += loss.item()
        num_trained += len(batch[self.data_key])
        lst_metrics.append(metrics)

    # average the metrics of the epoch (across batches)
    epoch_metrics = {k: sum([m[k] for m in lst_metrics]) / len(lst_metrics) for k in lst_metrics[0].keys()}

    return total_loss / num_trained, epoch_metrics
