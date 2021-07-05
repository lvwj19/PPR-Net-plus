import numpy as np
import torch

class PointCloudShuffle(object):
    def __init__(self, num_point):
        self.num_point = num_point

    def __call__(self, sample):
        pt_idxs = np.arange(0, self.num_point)
        np.random.shuffle(pt_idxs)

        sample['point_clouds'] = sample['point_clouds'][pt_idxs]
        sample['rot_label'] = sample['rot_label'][pt_idxs]
        sample['trans_label'] = sample['trans_label'][pt_idxs]
        sample['cls_label'] = sample['cls_label'][pt_idxs]
        if 'vis_label' in sample:
            sample['vis_label'] = sample['vis_label'][pt_idxs]
        
        return sample

class PointCloudJitter(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        all_noise = np.random.standard_normal(sample['point_clouds'].shape) * self.scale
        sample['point_clouds'] = sample['point_clouds'] + all_noise
        sample['point_clouds'] = sample['point_clouds'].astype(np.float32)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        sample['point_clouds'] = torch.from_numpy(sample['point_clouds'])
        sample['rot_label'] = torch.from_numpy(sample['rot_label'])
        sample['trans_label'] = torch.from_numpy(sample['trans_label'])
        sample['cls_label'] = torch.from_numpy(sample['cls_label'])
        if 'vis_label' in sample:
            sample['vis_label'] = torch.from_numpy(sample['vis_label'])

        return sample



    
