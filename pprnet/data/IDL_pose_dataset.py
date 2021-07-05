import os
import sys
FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.dirname(FILE_PATH)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE_PATH)))
sys.path.append(ROOT_DIR)

import pprnet.utils.dataset_util as dataset_util
import numpy as np
import torch
import torch.utils.data as data

class IDLPoseDataset(data.Dataset):
    def __init__(self, data_dir, load_ratio=1.0, mode='train',
                 transforms=None, scale=1000.0):
        self.num_point = 16384
        self.transforms = transforms
        if mode=='train':
            self.dataset, _ = dataset_util.load_dataset( \
                    data_dir, load_ratio,\
                    load_train_set=True, load_test_set=False)
        else:
            _, self.dataset = dataset_util.load_dataset( \
                    data_dir, load_ratio,\
                    load_train_set=False, load_test_set=True)
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
            # 'vis_label':self.dataset['vs_label'][idx].copy().astype(np.float32)
        }

        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample