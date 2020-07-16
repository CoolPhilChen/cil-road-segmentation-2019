from __future__ import division
import os.path as osp
import sys
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.modules.batchnorm import BatchNorm2d 

from config import config
from dataloader import get_train_loader
from network import CrfRnnNet
from datasets import Cil

from utils.init_func import init_weight, group_weight
from engine.engine import Engine

from network import CrfRnnNet

model = CrfRnnNet(2,n_iter=1)
ptr_model_pth = "log/snapshot/epoch-last.pth"
ptr_dict = torch.load(ptr_model_pth, map_location='cpu')['model']
model.load_state_dict(ptr_dict)

for name, param in model.crfrnn.named_parameters():
    if param.requires_grad:
        print(name, param.data)