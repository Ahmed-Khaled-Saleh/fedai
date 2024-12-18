"""Fill in a module description here"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/04_data.core.ipynb.

# %% auto 0
__all__ = ['FDblock']

# %% ../../nbs/04_data.core.ipynb 3
import numpy as np
import os
import sys
import random
import torch
import torchvision
from sklearn.model_selection import train_test_split
import ujson
import h5py
import torchvision.transforms as transforms
from fastcore.utils import *
from ..utils import *

# %% ../../nbs/04_data.core.ipynb 4
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# %% ../../nbs/04_data.core.ipynb 5
class FDblock:
    def __init__(self, cfg, partitioner: str):
        self.cfg = cfg
        self.config_path = self.cfg.data.dir_path + "config.json"
        self.train_path = self.cfg.data.dir_path + "train/"
        self.test_path = self.cfg.data.dir_path + "test/"

        self.partitioner_obj = get_class('fedai.data.partitioners', partitioner)  # noqa: F405
        self.dataidx_map = {}
        
        self.data = self.load_data(self.cfg.data.name)
        self.dataset_content, self.dataset_label = self.data if self.data is not None else (None, None)

# %% ../../nbs/04_data.core.ipynb 6
@patch
def check(self: FDblock):
    # check existing dataset
    if os.path.exists(self.config_path):
        with open(self.config_path, 'r') as f:
            config = ujson.load(f)
        
        if config['num_clients'] == self.cfg.num_clients and \
            config['non_iid'] == self.cfg.data.niid and \
            config['balance'] == self.cfg.data.balance and \
            config['partition'] == self.cfg.data.partition and \
            config['alpha'] == self.cfg.data.alpha and \
            config['batch_size'] == self.cfg.data.batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(self.train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.dirname(self.test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

# %% ../../nbs/04_data.core.ipynb 7
@patch
def load_data(self: FDblock):
    raise NotImplementedError

# %% ../../nbs/04_data.core.ipynb 8
@patch
def split_data(self: FDblock, X, y):
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
    # gc.collect()

    return train_data, test_data


# %% ../../nbs/04_data.core.ipynb 9
@patch
def partition(self: FDblock):
    # choose a paritioning method (iid, noniid, ext-nonidd) and split the data into train and test sets for all clients
    least_samples = int(min(self.cfg.data.batch_size / (1- self.cfg.data.train_ratio), len(self.dataset_label) / self.cfg.num_clients / 2))
    
    X, y, statistic = self.partitioner_obj.partition(self.dataset_content, self.dataset_label, least_samples, self.dataidx_map)
    
    for client in range(self.cfg.num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print("\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)
    
    del self.data

    train_data, test_data = self.split_data(X, y)
    return train_data, test_data

# %% ../../nbs/04_data.core.ipynb 10
@patch
def save_partitions(self: FDblock, cfg, train_data, test_data, statistic):
    """
    Save partitions using HDF5 format.
    
    Args:
        cfg: Configuration object.
        train_data: List of dictionaries for training data.
        test_data: List of dictionaries for test data.
        statistic: Statistical information to save.
    """
    config = {
        'num_clients': cfg.num_clients,
        'num_classes': cfg.data.num_classes,
        'non_iid': cfg.data.niid,
        'balance': cfg.data.balance,
        'partition': cfg.data.partition,
        'Size of samples for labels in clients': statistic,
        'alpha': self.cfg.data.alpha,
        'batch_size': self.cfg.data.batch_size,
    }

    print("Saving to disk in HDF5 format.\n")

    # Save training data
    with h5py.File(self.train_path + 'train_data.h5', 'w') as train_h5:
        for idx, train_dict in enumerate(train_data):
            group = train_h5.create_group(f'client_{idx}')
            for key, value in train_dict.items():
                group.create_dataset(key, data=value)

    # Save test data
    with h5py.File(self.test_path + 'test_data.h5', 'w') as test_h5:
        for idx, test_dict in enumerate(test_data):
            group = test_h5.create_group(f'client_{idx}')
            for key, value in test_dict.items():
                group.create_dataset(key, data=value)

    # Save configuration as a JSON file
    with open(self.config_path, 'w') as f:
        ujson.dump(config, f)


# %% ../../nbs/04_data.core.ipynb 11
@patch
def save_partitions_np(self: FDblock, cfg, train_data, test_data, statistic):
    
    config = {
        'num_clients': cfg.num_clients, 
        'num_classes': cfg.data.num_classes, 
        'non_iid': cfg.data.niid, 
        'balance': cfg.data.balance, 
        'partition': cfg.data.partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': self.cfg.data.alpha, 
        'batch_size': self.cfg.data.batch_size, 
    }

    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(self.train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)

    for idx, test_dict in enumerate(test_data):
        with open(self.test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
        
    with open(self.config_path, 'w') as f:
        ujson.dump(config, f)


# %% ../../nbs/04_data.core.ipynb 12
@patch
def tensorify(self: FDblock, data):
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    return {'x': X, 'y': y}

# %% ../../nbs/04_data.core.ipynb 13
@patch
def load_partition(self: FDblock, idx, split= 'train'):
    # loads the data for a client indexed by idx. 
    # By default it loads the training data but can also load the test data when split is set to 'test'
    data_dir = os.path.join(self.train_path) if split == 'train' else os.path.join(self.test_path)

    train_file = os.path.join(data_dir, str(idx) + '.npz')
    with open(train_file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    
    data_dict = self.tensorify(data)

    return data_dict

# %% ../../nbs/04_data.core.ipynb 14
@patch
def load_split(self: FDblock, split= 'train'):
    # load the data of all clients. By default, it loads the training data for all clients
    pass
