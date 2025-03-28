"""Fill in a module description here"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/07_vision.models.ipynb.

# %% auto 0
__all__ = ['MNISTCNNEncoder', 'MNISTCNNClassificationHead', 'MNISTCNN', 'CIFAR10Encoder', 'CIFAR10ClassificationHead',
           'CIFAR10Model']

# %% ../../nbs/07_vision.models.ipynb 3
from fastcore.utils import *  # noqa: F403
from torch import nn
import torch
from peft import *  # noqa: F403

# %% ../../nbs/07_vision.models.ipynb 5
class MNISTCNNEncoder(nn.Module):
    def __init__(self, in_channels=3, img_size=3, hidden_dim=512):
        super(MNISTCNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        final_size = img_size // 2 // 2
        self.fc1 = nn.Linear(64 * final_size * final_size, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x
    

# Classification Head Model
class MNISTCNNClassificationHead(nn.Module):
    def __init__(self, hidden_dim= 512, num_classes=10):
        super(MNISTCNNClassificationHead, self).__init__()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5) 

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    


class MNISTCNN(nn.Module):
    def __init__(self, in_channels=1, img_size= 28, hidden_dim=512, num_classes=10):
        super(MNISTCNN, self).__init__()
        self.encoder = MNISTCNNEncoder(in_channels=in_channels, img_size= img_size, hidden_dim=hidden_dim)
        self.classifier = MNISTCNNClassificationHead(hidden_dim= hidden_dim, num_classes= num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    

# %% ../../nbs/07_vision.models.ipynb 8
import torch
import torch.nn as nn
import numpy as np
import random

torch.manual_seed(42)  
np.random.seed(42)
random.seed(42)

# Encoder Model
class CIFAR10Encoder(nn.Module):
    def __init__(self, in_channels=3, img_size= 3, hidden_dim=512):
        super(CIFAR10Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Compute the flattened size dynamically at initialization
        final_size = img_size // 2 // 2 // 2  # 3 pool layers divide the size by 2 three times
        self.fc1 = nn.Linear(128 * final_size * final_size, hidden_dim)


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        return x

# Classification Head Model
class CIFAR10ClassificationHead(nn.Module):
    def __init__(self, hidden_dim= 512, num_classes=10):
        super(CIFAR10ClassificationHead, self).__init__()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5) 

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CIFAR10Model(nn.Module):
    def __init__(self, in_channels=3, img_size= 3, hidden_dim=512, num_classes=10):
        super(CIFAR10Model, self).__init__()
        self.encoder = CIFAR10Encoder(in_channels=in_channels, img_size= img_size, hidden_dim=hidden_dim)
        self.classifier = CIFAR10ClassificationHead(hidden_dim= hidden_dim, num_classes= num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

