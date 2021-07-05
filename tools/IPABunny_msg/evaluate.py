import os
import sys
FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.dirname(FILE_PATH)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE_PATH)))
print(FILE_PATH)
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import time
import random
from pprnet.pprnet import PPRNet, load_checkpoint
from pprnet.object_type import ObjectType
import pprnet.utils.eulerangles as eulerangles
import pprnet.utils.eval_util as eval_util
import pprnet.utils.visualize_util as visualize_util
from pprnet.data.IPA_pose_dataset import IPAPoseDataset
from pprnet.data.pointcloud_transforms import PointCloudShuffle, ToTensor, PointCloudJitter
from torchvision import transforms
from torch.utils.data import DataLoader

import pprnet.utils.show3d_balls as show3d_balls

#-----------------------GLOBAL SETTINGS START-----------------------
MODEL_OBJ_DIRS = ['/home/dongzhikai/Desktop/iros_competition_code/CAD/SileaneBunny.obj']
BATCH_SIZE = 1
NUM_TYPE = 1
NUM_POINT = 16384
CHECKPOINT_PATH = "../../logs/IPABunny_msg/log1_batch8_scale3_continue/checkpoint.tar"
DATASET_DIR = 'your-path/Fraunhofer_IPA_Bin-Picking_dataset/h5_dataset/bunny/train/'
TEST_CYCLE_RANGE = [499,500]
TEST_SCENE_RANGE = [1, 81]
#-----------------------GLOBAL SETTINGS END-----------------------

#-----------------------------------------------------------------
model_point_clouds = [ eval_util.extract_vertexes_from_obj(path)*1000.0 for path in MODEL_OBJ_DIRS ]
transforms = transforms.Compose(
    [
        PointCloudShuffle(NUM_POINT),
        ToTensor()
    ]
)
test_dataset = IPAPoseDataset(DATASET_DIR, TEST_CYCLE_RANGE, TEST_SCENE_RANGE, transforms=transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# build net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
type_bunny = ObjectType(type_name='bunny', class_idx=0, symmetry_type='finite',
                 lambda_p=[[0.0263663, 0.0, 0.0], [0.0, 0.0338224, 0.0], [-0.0, 0.0, 0.0484393]],
                 G=[ [[1,0,0], [0,1,0], [0,0,1]] ])
backbone_config = {
    'npoint_per_layer': [4096,1024,256,64],
    'radius_per_layer': [[10,20,30],[30,45,60],[60,80,120],[120,160,240]]
}
net = PPRNet(type_bunny, 'pointnet2msg', backbone_config, True, None, False)
net.to(device)
net, _, _ = load_checkpoint(CHECKPOINT_PATH, net)
#-----------------------------------------------------------------



def eval_one_epoch():
    net.eval() #

    for batch_idx, batch_samples in enumerate(test_loader):
        # labels = {
        #     'rot_label':batch_samples['rot_label'].to(device),
        #     'trans_label':batch_samples['trans_label'].to(device),
        # }
        input_point = batch_samples['point_clouds'][0].cpu().numpy().copy()
        inputs = {
            'point_clouds': batch_samples['point_clouds'].to(device),
            'labels': None
        }

        # Forward pass
        with torch.no_grad():
            pred_results, _ = net(inputs)

        pred_trans_val = pred_results[0][0].cpu().numpy()
        pred_mat_val = pred_results[1][0].cpu().numpy()
        if pred_results[3] is not None:
            pred_cls_val = pred_results[3][0].cpu().numpy()
            pred_cls_val = np.argmax(pred_cls_val, -1)
        else:
            pred_cls_val = np.zeros(pred_trans_val.shape[0])


        pc_list = []
        trans_list = []
        rot_list = []
        cls_list = []
        for k in range(NUM_TYPE):
            cls_k_idx = np.where(pred_cls_val==k)[0]
            if len(cls_k_idx) == 0:
                continue
            cls_k_points = input_point[cls_k_idx]
            cls_k_pred_trans = pred_trans_val[cls_k_idx]
            cls_k_pred_mat = pred_mat_val[cls_k_idx]

            meanshift_args = {'bandwidth':40, 'bin_seeding':True, 'cluster_all':False, 'min_bin_freq':40}
            n_cluster_cls_k, ins_label_cls_k, centroid_cls_k, rot_mat_cls_k, pc_segments_cls_k =  \
                eval_util.cluster_and_average(cls_k_points, cls_k_pred_trans, cls_k_pred_mat, meanshift_args)

            pc_list += pc_segments_cls_k
            trans_list += centroid_cls_k
            rot_list += rot_mat_cls_k
            cls_list += [k]*len(pc_segments_cls_k)
        
        visualize_util.show_points([input_point], radius=5)
        visualize_util.show_points(pc_list, color_array='random', radius=5)
        visualize_util.show_models(model_point_clouds, trans_list, rot_list, cls_list, color_array='random', radius=5)


if __name__ == "__main__":
    eval_one_epoch()
