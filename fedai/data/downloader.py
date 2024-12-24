"""Fill in a module description here"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/04_data.downloader.ipynb.

# %% auto 0
__all__ = ['BaseDownloader', 'LLMDataCollator']

# %% ../../nbs/04_data.downloader.ipynb 3
import numpy as np
import os
import random
import torch
from sklearn.model_selection import train_test_split
import ujson
import h5py
from fastcore.utils import *
from ..utils import *

# %% ../../nbs/04_data.downloader.ipynb 4
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# %% ../../nbs/04_data.downloader.ipynb 5
class BaseDownloader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.config_path = os.path.join(self.cfg.data.dir_path, self.cfg.data.name , "config.json")
        self.train_path = os.path.join(self.cfg.data.dir_path, self.cfg.data.name , "train")
        self.test_path = os.path.join(self.cfg.data.dir_path, self.cfg.data.name, "test")
        print(self.train_path, self.test_path)
        self.partitioner_obj = get_class('fedai.data.partitioners', self.cfg.data.partitioner)(self.cfg) # noqa: F405
        self.dataidx_map = {}
        
        self.data = self.load_data()
        self.dataset_content, self.dataset_label = self.data if self.data is not None else (None, None)

# %% ../../nbs/04_data.downloader.ipynb 6
@patch
def check(self: BaseDownloader):
    # check existing dataset
    if os.path.exists(self.config_path):
        with open(self.config_path, 'r') as f:
            config = ujson.load(f)
        
        if config['num_clients'] == self.cfg.num_clients and \
            config['non_iid'] == self.cfg.data.niid and \
            config['balance'] == self.cfg.data.balance and \
            config['partition'] == self.cfg.data.partitioner and \
            config['alpha'] == self.cfg.data.alpha and \
            config['batch_size'] == self.cfg.data.batch_size:
            print("\nDataset already generated.\n")
            return True

    print(f"\nDataset not found, Downloading the dataset: {self.cfg.data.name}.\n")
   
    if not os.path.exists(self.train_path):
        os.makedirs(self.train_path)

    if not os.path.exists(self.test_path):
        os.makedirs(self.test_path)

    return False

# %% ../../nbs/04_data.downloader.ipynb 7
@patch
def load_data(self: BaseDownloader):
    raise NotImplementedError

# %% ../../nbs/04_data.downloader.ipynb 8
@patch
def split_data(self: BaseDownloader, X, y):
    # Split dataset into train and test sets
    # make a list of train and test data where each element is a dictionary of x and y
    # each element in the list is a client's data
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size= self.cfg.data.train_ratio, shuffle= True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y

    return train_data, test_data


# %% ../../nbs/04_data.downloader.ipynb 9
@patch
def partition(self: BaseDownloader):
    # choose a paritioning method (iid, noniid, ext-nonidd) and split the data into train and test sets for all clients
    least_samples = int(min(self.cfg.data.batch_size / (1- self.cfg.data.train_ratio), len(self.dataset_label) / self.cfg.num_clients / 2))
    X, y, statistic = self.partitioner_obj.partition(self.dataset_content, self.dataset_label, least_samples, self.dataidx_map)
    
    for client in range(self.cfg.num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print("\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)
    
    del self.data

    train_data, test_data = self.split_data(X, y)
    return train_data, test_data, statistic

# %% ../../nbs/04_data.downloader.ipynb 10
@patch
def save_partitions(self: BaseDownloader, train_data, test_data, statistic):
    """
    Save partitions using HDF5 format.
    
    Args:
        cfg: Configuration object.
        train_data: List of dictionaries for training data.
        test_data: List of dictionaries for test data.
        statistic: Statistical information to save.
    """
    config = {
        'num_clients': self.cfg.num_clients,
        'num_classes': self.cfg.data.num_classes,
        'non_iid': self.cfg.data.niid,
        'balance': self.cfg.data.balance,
        'partition': self.cfg.data.partitioner,
        'Size of samples for labels in clients': statistic,
        'alpha': self.cfg.data.alpha,
        'batch_size': self.cfg.data.batch_size,
    }

    print("Saving to disk in HDF5 format.\n")

    # Save training data
    with h5py.File(os.path.join(self.train_path, 'train_data.h5'), 'w') as train_h5:
        for idx, train_dict in enumerate(train_data):
            group = train_h5.create_group(f'client_{idx}')
            for key, value in train_dict.items():
                group.create_dataset(key, data=value)

    # Save test data
    with h5py.File(os.path.join(self.test_path, 'test_data.h5'), 'w') as test_h5:
        for idx, test_dict in enumerate(test_data):
            group = test_h5.create_group(f'client_{idx}')
            for key, value in test_dict.items():
                group.create_dataset(key, data=value)

    # Save configuration as a JSON file
    with open(self.config_path, 'w') as f:
        ujson.dump(config, f)

    self.save_space(train_data, test_data, statistic)

# %% ../../nbs/04_data.downloader.ipynb 11
@patch
def save_space(self: BaseDownloader, train_data, test_data, statistic):
    import gc
    del self.dataset_content
    del self.dataset_label
    del train_data
    del test_data
    del statistic
    gc.collect()
    

# %% ../../nbs/04_data.downloader.ipynb 13
class LLMDataCollator:
    pass
