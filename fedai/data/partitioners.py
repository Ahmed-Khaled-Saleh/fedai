"""Fill in a module description here"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/04_data.partitioners.ipynb.

# %% auto 0
__all__ = ['BasePartitioner']

# %% ../../nbs/04_data.partitioners.ipynb 3
import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from fastcore.utils import *

# %% ../../nbs/04_data.partitioners.ipynb 4
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# %% ../../nbs/04_data.partitioners.ipynb 5
class BasePartitioner:
    def __init__(self, cfg):
        self.cfg = cfg

    def assign(self, dataset_content, dataset_label, dataidx_map, X, y, statistic):
        for client in range(self.cfg.num_clients):
            idxs = dataidx_map[client]
            X[client] = dataset_content[idxs]
            y[client] = dataset_label[idxs]
            for i in np.unique(y[client]):
                statistic[client].append((int(i), int(sum(y[client]==i))))

# %% ../../nbs/04_data.partitioners.ipynb 6
@patch
def parition(self: BasePartitioner, dataset_content, dataset_label, least_samples, dataidx_map):
    "The traditional IID partitioning method"
    X = [[] for _ in range(self.cfg.num_clients)]
    y = [[] for _ in range(self.cfg.num_clients)]
    statistic = [[] for _ in range(self.cfg.num_clients)]
    
    idxs = np.array(range(len(dataset_label)))
    idx_for_each_class = []
    for i in range(self.cfg.data.num_classes):
        idx_for_each_class.append(idxs[self.cfg.data.dataset_label == i])

    class_num_per_client = [self.cfg.data.class_per_client for _ in range(self.cfg.num_clients)]
    for i in range(self.cfg.num_classes):
        selected_clients = []
        for client in range(self.cfg.num_clients):
            if class_num_per_client[client] > 0:
                selected_clients.append(client)
        if len(selected_clients) == 0:
            break
        selected_clients = selected_clients[:int(np.ceil((self.cfg.num_clients/self.cfg.data.num_classes)*self.cfg.data.class_per_client))]

        num_all_samples = len(idx_for_each_class[i])
        num_selected_clients = len(selected_clients)
        num_per = num_all_samples / num_selected_clients
        if self.cfg.balance:
            num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
        else:
            num_samples = np.random.randint(max(num_per/10, least_samples/self.cfg.num_classes), num_per, num_selected_clients-1).tolist()
        num_samples.append(num_all_samples-sum(num_samples))

        idx = 0
        for client, num_sample in zip(selected_clients, num_samples):
            if client not in dataidx_map.keys():
                dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
            else:
                dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
            idx += num_sample
            class_num_per_client[client] -= 1

    X, y, statistic = self.assign(dataset_content, dataset_label, dataidx_map, X, y, statistic)

    return X, y, statistic
