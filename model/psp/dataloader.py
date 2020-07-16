import cv2
import torch
import numpy as np
from torch.utils import data

from config import config
from utils.img_utils import random_scale, random_mirror, normalize, \
    generate_random_crop_pos, random_crop_pad_to_shape

def img_to_black(img, threshold=50):
    """Helper function to binarize greyscale images with a cut-off."""
    img = img.astype(np.int64)
    idx = img[:, :] > threshold
    idx_0 = img[:, :] <= threshold
    img[idx] = 1
    img[idx_0] = 0
    return img

class TrainPre(object):
    """Data Pre-processing."""
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, img, gt):
        img, gt = random_mirror(img, gt) # images are randomly flipped to increase variance

        gt = img_to_black(gt) # binary filter on gt.

        if config.train_scale_array is not None:
            img, gt, scale = random_scale(img, gt, config.train_scale_array) # scale the images with supplied list

        img = normalize(img, self.img_mean, self.img_std)

        crop_size = (200, 200)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size) # obtain random location

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0) # get the cropped images and re-sized to crop-size
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)

        p_img = cv2.resize(p_img, (config.image_width // config.gt_down_sampling,
                                 config.image_height // config.gt_down_sampling),
                          interpolation=cv2.INTER_NEAREST) # resize by downsampling

        p_gt = cv2.resize(p_gt, (config.image_width // config.gt_down_sampling,
                                 config.image_height // config.gt_down_sampling),
                          interpolation=cv2.INTER_NEAREST)

        p_img = p_img.transpose(2, 0, 1)

        extra_dict = None

        return p_img, p_gt, extra_dict

class TrainPreOri(TrainPre):
    """No pre-processing"""

    def __call__(self, img, gt):
        (img, gt) = random_mirror(img, gt)
        gt = img_to_black(gt)
        if config.train_scale_array is not None:
            (img, gt, scale) = random_scale(img, gt,
                    config.train_scale_array)
        img = normalize(img, self.img_mean, self.img_std)
        (p_img, p_gt) = (img, gt)

        p_img = cv2.resize(p_img, (config.image_width // config.gt_down_sampling,
                           config.image_height // config.gt_down_sampling),
                           interpolation=cv2.INTER_NEAREST)
        p_gt = cv2.resize(p_gt, (config.image_width // config.gt_down_sampling,
                           config.image_height // config.gt_down_sampling),
                           interpolation=cv2.INTER_NEAREST)
        p_img = p_img.transpose(2, 0, 1)
        extra_dict = None
        return (p_img, p_gt, extra_dict)


def get_train_loader(engine, dataset):
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'test_source': config.test_source}

    train_preprocess = TrainPre(config.image_mean, config.image_std)
    train_preprocess_no_crop = TrainPreOri(config.image_mean,
            config.image_std)
    # train set with pre-processing
    train_dataset = dataset(data_setting, "train", train_preprocess,
                            config.batch_size * config.niters_per_epoch)
    # train set without pre-processing
    train_dataset_no_crop = dataset(data_setting, 'train', train_preprocess_no_crop,
                                    config.batch_size * config.niters_per_epoch)
    # combine two dataset
    train_dataset = data.ConcatDataset([train_dataset,
            train_dataset_no_crop])

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size
    
    # add distributed support
    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler

