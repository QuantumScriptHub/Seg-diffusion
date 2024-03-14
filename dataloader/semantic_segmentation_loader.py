from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.image_util import read_text_lines, resize_max_res


class SemanticSegmentationDataset(Dataset):
    def __init__(self,
                 train_img_list,
                 train_gt_list,
                 train_text_list,
                 processing_res=384,
                 transform=None):
        super(SemanticSegmentationDataset, self).__init__()

        self.transform = transform
        self.processing_res = processing_res
        self.samples = []
        lines_img = read_text_lines(train_img_list)
        lines_gt = read_text_lines(train_gt_list)
        with open(train_text_list, 'r') as f:
            prompt = json.load(f)

        for idx in range(len(lines_img)):
            sample = dict()
            sample['original'] = lines_img[idx]
            sample['gt'] = lines_gt[idx]
            sample['prompt'] = prompt[os.path.basename(lines_img[idx])]
            self.samples.append(sample)

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]
        original = Image.open(sample_path['original'])
        original = resize_max_res(original, self.processing_res)
        sample['original'] = np.array(original.convert('RGB')).astype(np.float32)
        gt = Image.open(sample_path['gt'])
        gt = resize_max_res(gt, self.processing_res)
        tmp = np.array(gt).astype(np.float32)
        if len(tmp.shape) == 2:
            tmp = np.expand_dims(tmp, 2)
        else:
            tmp = tmp[:, :, 0:1]
        sample['gt'] = np.repeat(tmp, 3, 2)

        if self.transform is not None:
            sample = self.transform(sample)
        sample['prompt'] = sample_path['prompt']

        return sample

    def __len__(self):
        return len(self.samples)
