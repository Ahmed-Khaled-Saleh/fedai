"""Fill in a module description here"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/04_vision.data.ipynb.

# %% auto 0
__all__ = []

# %% ../../nbs/04_vision.data.ipynb 3
import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from ..data.core import *
from ..data import *
from ..utils import *
from fastcore.utils import *

# %% ../../nbs/04_vision.data.ipynb 4
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# %% ../../nbs/04_vision.data.ipynb 6
@patch
def load_data(self: FVDownloader):
    if not os.path.exists(self.cfg.data.dir_path):
        os.makedirs(self.cfg.data.dir_path)
    
    ds_class = get_class(torchvision.datasets, self.cfg.data.name)  # noqa: F405
    # Setup directory for train/test data

    if check(self.config_path, self.train_path, self.test_path, self.cfg.num_clients, self.cfg.data.niid, self.cfg.data.balance, self.cfg.data.partition):
        return
    
    transform = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize(self.transform_norm_mapping[self.cfg.data.name][0],
                             self.transform_norm_mapping[self.cfg.data.name][1])])

    trainset = ds_class(
        root=self.cfg.data.dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = ds_class(
        root=self.cfg.data.dir_path+"rawdata", train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    return dataset_image, dataset_label
