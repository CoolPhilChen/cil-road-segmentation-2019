#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import argparse
import numpy as np

import torch
import torch.multiprocessing as mp

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from engine.evaluator import Evaluator
from engine.logger import get_logger
from seg_opr.metric import hist_info, compute_score
from tools.benchmark import compute_speed, stat
from datasets import Cil
from network import PSPNet

logger = get_logger()

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        name = data['fn']

        img = cv2.resize(img, (config.image_height,config.image_width),
                         interpolation=cv2.INTER_LINEAR)
        # predicted labels
        pred = self.whole_eval(img,
                               (config.image_height // config.gt_down_sampling,
                                config.image_width // config.gt_down_sampling),
                               device)
        pred.astype(np.float32)

        pred = cv2.resize(pred, (config.test_image_height,config.test_image_width),
                         interpolation=cv2.INTER_NEAREST)

        # binary to greyscale
        pred = 237 * pred

        results_dict = {'rmse': 1}

        if self.save_path is not None:
            fn = name + '.png'
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            # logger.info("Save the image " + fn)

        return results_dict

    def compute_metric(self, results):
        """No metric is calculated during prediction."""
        return 'no result'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--input_size', type=str, default='1x3x608x608',
                        help='Input size. '
                             'channels x height x width (default: 1x3x224x224)')
    parser.add_argument('-speed', '--speed_test', action='store_true')
    parser.add_argument('--iteration', type=int, default=5000)
    parser.add_argument('-summary', '--summary', action='store_true')
    
    parser.add_argument('-op', '--output_prob', default=False, action='store_true')
    parser.add_argument('-m', '--mode', default='test', type=str)

    args = parser.parse_args()
    # all_dev = parse_devices(args.devices) #gpu
    all_dev = [0]   #cpu

    network = PSPNet(config.num_classes)
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'test_source': config.test_source}
    dataset = Cil(data_setting, args.mode, None)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                 config.image_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path, args.output_prob)
        segmentor.run(config.snapshot_dir, args.epochs, config.val_log_file,
                          config.link_val_log_file)

