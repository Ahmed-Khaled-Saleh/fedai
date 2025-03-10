"""Fill in a module description here"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/04_vision.VisionBlock.ipynb.

# %% auto 0
__all__ = ['VisionBlock']

# %% ../../nbs/04_vision.VisionBlock.ipynb 3
import numpy as np
import os
import sys
import random
import gc
import h5py
import torch
import torchvision
from sklearn.model_selection import train_test_split
import ujson
import torchvision.transforms as transforms
from fastcore.utils import *
from .downloader import *

# %% ../../nbs/04_vision.VisionBlock.ipynb 4
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# %% ../../nbs/04_vision.VisionBlock.ipynb 5
class VisionBlock(torch.utils.data.Dataset):
    def __init__(self, cfg, id, train= True, download= True, transform=None):
        self.cfg = cfg
        self.config_path = os.path.join(self.cfg.data.data_dir, self.cfg.data.name , "config.json")
        self.train_path = os.path.join(self.cfg.data.data_dir, self.cfg.data.name , "train")
        self.test_path = os.path.join(self.cfg.data.data_dir, self.cfg.data.name, "test")
        self.train = train
        self.transform = transform
        self.id = id

        if download:
            self.download_data()
        
    def download_data(self):
        self.downloader = VisionDownloader(self.cfg, self.transform)
        if isinstance(self.downloader.dataset_content, (list, tuple, np.ndarray)):
            train_data, test_data, stats = self.downloader.partition()
            print('saving')
            self.downloader.save_partitions(train_data, test_data, stats)
            del self.downloader
            gc.collect()
            
    def tensorify(self, data):
        X = torch.tensor(data['x'], dtype= torch.float32)
        y = torch.tensor(data['y'], dtype= torch.int64)
        return {'x': X, 'y': y}

    def load_single_client_data(self, idx):
        path, dir = (self.train_path, 'train') if self.train else (self.test_path, 'test')

        with h5py.File(os.path.join(path, f'{dir}_data.h5'), 'r') as hf_file:
            client_data = hf_file[f'client_{self.id}']
            data = {key: client_data[key][idx] for key in client_data.keys()}
            data = self.tensorify(data)
        return data
    
    def __getitem__(self, idx):
        if idx < 0: # manage minus idx
            idx = len(self) + idx
        return self.load_single_client_data(idx)
    
    def __len__(self):
        path, dir = (self.train_path, 'train') if self.train else (self.test_path, 'test')
        with h5py.File(os.path.join(path, f'{dir}_data.h5'), 'r') as hf_file:
            client_data = hf_file[f'client_{self.id}']
            return len(client_data['x'])
