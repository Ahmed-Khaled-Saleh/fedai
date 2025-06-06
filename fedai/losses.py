"""ِAny custom losses should be here"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/11_losses.ipynb.

# %% auto 0
__all__ = ['AnchorLoss']

# %% ../nbs/11_losses.ipynb 3
import copy
import numpy as np
import torch, os, random
import torch.nn as nn
import torch.nn.functional as F
from fastcore.utils import * # type: ignore # noqa: F403
from copy import deepcopy

# %% ../nbs/11_losses.ipynb 4
class AnchorLoss(nn.Module):
    def __init__(self, random_seed, num_classes, feature_num, t=1, h_c=None):
        super().__init__()
        self.num_classes = num_classes
        self.feature_num = feature_num

        torch.manual_seed(random_seed)  
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.anchor = nn.Parameter(F.normalize(torch.randn(num_classes, feature_num)), requires_grad=True)
        
        if t > 1 and h_c is not None:
            print("Updating anchor with h_c")
            with torch.no_grad():  # This ensures the operation doesn't track gradients
                h_c = h_c.to(self.anchor.device)
                self.anchor.copy_(h_c)
        # self.anchor = nn.Parameter(h_c, requires_grad= True) if t > 1 and h_c  else self.anchor

# %% ../nbs/11_losses.ipynb 5
@patch
def forward(self: AnchorLoss, feature, _target, Lambda = 0.1):
    assert not torch.isnan(_target).any(), "Found NaN in _target!"
    # broadcast feature anchors for all inputs
    centre = self.anchor.cuda().index_select(dim=0, index=_target.long())
    # compute the number of samples in each class
    counter = torch.histc(_target.cpu().float(), bins=self.num_classes, min=0, max=self.num_classes-1)
    counter = counter.to(_target.device)  # Move back to the same device as _target
    count = counter[_target.long()]
    centre_dis = feature - centre				# compute distance between input and anchors
    pow_ = torch.pow(centre_dis, 2)				# squre
    sum_1 = torch.sum(pow_, dim=1)
    count = count.clamp(min=1)  # Avoid division by zero
    dis_ = sum_1 / count.float()				# sum all distance
    # dis_ = torch.div(sum_1, count.float())		# mean by class
    sum_2 = torch.sum(dis_)/self.num_classes						# mean loss
    res = Lambda*sum_2   							# time hyperparameter lambda 
    return res
