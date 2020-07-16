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
from network import PSPNet
from datasets import Cil

from utils.init_func import init_weight, group_weight
from engine.engine import Engine
from engine.lr_policy import PolyLR


parser = argparse.ArgumentParser()

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True

    seed = config.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, Cil)
    # load model    
    criterion = nn.CrossEntropyLoss(reduction='mean',
                                    ignore_index=-1)
    model = PSPNet(config.num_classes, criterion=criterion,
               pretrained_model=config.pretrained_model,
               norm_layer=BatchNorm2d)
    base_lr = config.lr
    init_weight(model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    params_list = []
    params_list = group_weight(params_list, model.backbone,
                               BatchNorm2d, base_lr)
    for module in model.business_layer:
        params_list = group_weight(params_list, module, BatchNorm2d,
                                   base_lr * 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)
    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
    # for state in optimizer.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             state[k] = v.cuda()
    # optimizer = torch.optim.Adam(model.parameters())
    # register state dictations
    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()
    # start training process
    for epoch in range(engine.state.epoch, config.nepochs):
        # if engine.distributed:
        #     train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        for idx in pbar:
            optimizer.zero_grad()
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']

            if torch.cuda.is_available():
                imgs = imgs.cuda(non_blocking=True)
                gts = gts.cuda(non_blocking=True)
            

            loss = model(imgs, gts)
            loss.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]['lr'] 
            # current_idx = epoch * config.niters_per_epoch + idx
            # lr  = lr_policy.get_lr(current_idx)
            # optimizer.param_groups[0]['lr'] = lr
            # optimizer.param_groups[1]['lr'] = lr
            # for i in range(2, len(optimizer.param_groups)):
            #     optimizer.param_groups[i]['lr'] = lr * 10

            # print information
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % loss

            pbar.set_description(print_str, refresh=False)

        if (epoch%5==0):
            engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            
