import os
import sys
FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.dirname(FILE_PATH)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE_PATH)))
# print(FILE_PATH)
# print(ROOT_DIR)
# exit()
sys.path.append(ROOT_DIR)

import pprnet.utils.dataset_util as dataset_util
import numpy as np
import torch
import torch.utils.data as data

class IPAPoseDataset(data.Dataset):
    def __init__(self, data_dir, cycle_range, scene_range, mode='train',
                 transforms=None, collect_names=False, collect_error_names=False, scale=1000.0):
        self.num_point = 16384
        self.transforms = transforms
        self.collect_names = collect_names
        self.dataset = dataset_util.load_dataset_by_cycle( \
                data_dir, range(cycle_range[0], cycle_range[1]), range(scene_range[0], scene_range[1]),\
                mode, collect_names, collect_error_names)
        # convert to mm
        self.dataset['data'] *= scale
        self.dataset['trans_label'] *= scale

    def __len__(self):
        return self.dataset['data'].shape[0]

    def __getitem__(self, idx):
        sample = {
            'point_clouds': self.dataset['data'][idx].copy().astype(np.float32),
            'rot_label': self.dataset['rot_label'][idx].copy().astype(np.float32),
            'trans_label':self.dataset['trans_label'][idx].copy().astype(np.float32),
            'cls_label':self.dataset['cls_label'][idx].copy().astype(np.int64),
            'vis_label':self.dataset['vs_label'][idx].copy().astype(np.float32)
        }
        if self.collect_names:
            sample['name'] = self.dataset['name'][idx]

        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample
