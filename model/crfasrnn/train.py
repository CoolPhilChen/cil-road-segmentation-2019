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
                                    ignore_index=255)
    model = CrfRnnNet(config.num_classes, criterion=criterion,
               pretrained_model=config.pretrained_model,
               norm_layer=BatchNorm2d, n_iter=5)
    #`n_iter=5`: during training, we set the number of mean-field iterations T in the CRF-RNN to 5    

    # load pretrained weights
    ptr_model_pth = "/home/philchen/cil-road-segmentation-2019/log/psp/snapshot/epoch-10.pth"
    ptr_dict = torch.load(ptr_model_pth, map_location='cpu')['model']
    ptr_dict = {k: v for k, v in ptr_dict.items() if k in model.psp.state_dict()}
    model.psp.load_state_dict(ptr_dict)

    # randomly initialize parameters
    # init_weight(model.psp.business_layer, nn.init.kaiming_normal_,
    #             BatchNorm2d, config.bn_eps, config.bn_momentum,
    #             mode='fan_in', nonlinearity='relu')

    
    # group weight initialization on all layers
    # Q: why use group_weight?
    params_list = []
    params_list = group_weight(params_list, model.psp.backbone,
                               BatchNorm2d, config.lr)
    for module in model.psp.business_layer:
        params_list = group_weight(params_list, module, BatchNorm2d,
                                   config.lr)
    params_list.append(
        dict(params=list(model.crfrnn.parameters()), weight_decay=config.weight_decay, lr=config.lr)) #todo

    # optimizer = torch.optim.Adam(params_list)
    optimizer = torch.optim.SGD(params_list,
                                lr=config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # register state dictations
    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    print("training start...")  #debug
    model.train()
    # start training process
    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
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

            lr = optimizer.param_groups[0]['lr'] 

            loss.backward()
            optimizer.step()
            # print information
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % loss

            pbar.set_description(print_str, refresh=False)

        if (epoch%10==0):
            engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            
