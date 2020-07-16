import os
import os.path as osp
import cv2
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from engine.logger import get_logger
from utils.pyt_utils import load_model, link_file, ensure_dir
from utils.img_utils import pad_image_to_shape, normalize

logger = get_logger()

# Evaluator class manage whole evaluation process with distributed processing
# and non-distributed processing
class Evaluator(object):
    def __init__(self, dataset, class_num, image_mean, image_std, network,
                 multi_scales, is_flip, devices,
                 verbose=False, save_path=None, output_prob=True):
        self.dataset = dataset
        self.ndata = self.dataset.get_length()
        self.class_num = class_num
        self.image_mean = image_mean
        self.image_std = image_std
        self.multi_scales = multi_scales
        self.is_flip = is_flip
        self.network = network
        self.devices = devices

        self.context = mp.get_context('spawn')
        self.val_func = None
        self.results_queue = self.context.Queue(self.ndata)

        self.verbose = verbose
        self.save_path = save_path
        if save_path is not None:
            ensure_dir(save_path)

        self.output_prob = output_prob

    # Function for running evaluation process
    def run(self, model_path, model_indice, log_file, log_file_link):
        """Evaluate models."""
        if '.pth' in model_indice:
            models = [model_indice, ]
        else:
            models = [os.path.join(model_path,
                                   'epoch-%s.pth' % model_indice), ]

        results = open(log_file, 'a')
        link_file(log_file, log_file_link)

        for model in models:
            logger.info("Load Model: %s" % model)
            self.val_func = load_model(self.network, model)
            result_line = self.multi_process_evaluation()

            results.write('Model: ' + model + '\n')
            results.write(result_line)
            results.write('\n')
            results.flush()

        results.close()

    # multi-device distributed processing if the dataset is too large
    def multi_process_evaluation(self):
        start_eval_time = time.perf_counter()
        nr_devices = len(self.devices)
        stride = int(np.ceil(self.ndata / nr_devices))

        # start multi-process on multi-gpu
        procs = []
        for d in range(nr_devices):
            e_record = min((d + 1) * stride, self.ndata)
            shred_list = list(range(d * stride, e_record))
            device = self.devices[d]
            logger.info(
                'GPU %s handle %d data.' % (device, len(shred_list)))
            p = self.context.Process(target=self.worker,
                                     args=(shred_list, device))
            procs.append(p)

        for p in procs:
            p.start()

        all_results = []
        for _ in tqdm(range(self.ndata)):
            t = self.results_queue.get()
            all_results.append(t)
            if self.verbose:
                self.compute_metric(all_results)

        for p in procs:
            p.join()

        result_line = self.compute_metric(all_results)
        logger.info(
            'Evaluation Elapsed Time: %.2fs' % (
                    time.perf_counter() - start_eval_time))
        return result_line

    def worker(self, shred_list, device):
        start_load_time = time.time()
        logger.info('Load Model on Device %d: %.2fs' % (
            device, time.time() - start_load_time))
        for idx in shred_list:
            dd = self.dataset[idx]
            results_dict = self.func_per_iteration(dd, device)
            self.results_queue.put(results_dict)

    def func_per_iteration(self, data, device):
        raise NotImplementedError
    
    # inherited from eval.py 
    def compute_metric(self, results):
        raise NotImplementedError

    # function for evaluating the whole image at once and select the most 
    # probable prediction amongst all. 
    def whole_eval(self, img, output_size, input_size=None, device=None):
        if input_size is not None:
            img, margin = self.process_image(img, input_size)
        else:
            img = self.process_image(img, input_size)

        pred = self.val_func_process(img, device)
        if input_size is not None:
            pred = pred[:, margin[0]:(pred.shape[1] - margin[1]),
                   margin[2]:(pred.shape[2] - margin[3])]
        pred = pred.permute(1, 2, 0)
        pred = pred.cpu().numpy()
        if output_size is not None:
            pred = cv2.resize(pred,
                              (output_size[1], output_size[0]),
                              interpolation=cv2.INTER_LINEAR)

        if self.output_prob:
            pred = pred[:, :, 1]
        else:
            pred = pred.argmax(2)


        return pred

    # function for loading image into model and form prediction 
    def val_func_process(self, input_data, device=None):
        input_data = np.ascontiguousarray(input_data[None, :, :, :],
                                          dtype=np.float32)
        # input_data = torch.FloatTensor(input_data).cuda(device)   #gpu
        input_data = torch.FloatTensor(input_data)

        # with torch.cuda.device(input_data.get_device()):  #gpu
        self.val_func.eval()
        # self.val_func.to(input_data.get_device())
        with torch.no_grad():
            score = self.val_func(input_data)
            score = score[0]

            if self.is_flip:
                input_data = input_data.flip(-1)
                score_flip = self.val_func(input_data)
                score_flip = score_flip[0]
                score += score_flip.flip(-1)
            score = torch.exp(score)
            # score = score.data

        return score
    
    # function for input image munipulation to correct dimension. 
    def process_image(self, img, crop_size=None):
        p_img = img

        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), axis=2)

        p_img = normalize(p_img, self.image_mean, self.image_std)

        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size,
                                               cv2.BORDER_CONSTANT, value=0)
            p_img = p_img.transpose(2, 0, 1)

            return p_img, margin

        p_img = p_img.transpose(2, 0, 1)

        return p_img
