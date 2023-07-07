import os
import sys

FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.dirname(FILE_PATH)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE_PATH)))
print(FILE_PATH)
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
# from pprnet.pprnet import PPRNet, load_checkpoint
from pprnet.pprnet_plus import PPRNetPlus, load_checkpoint, save_checkpoint
from pprnet.object_type import ObjectType

import math
from datetime import datetime
import h5py
import numpy as np
import importlib
import torch
import torch.nn as nn

import time
import pprnet.utils.show3d_balls as show3d_balls

import sklearn
from sklearn.cluster import MeanShift
import random
import pprnet.utils.eulerangles as eulerangles
from pprnet.data.IPA_pose_dataset import IPAPoseDataset
from pprnet.data.pointcloud_transforms import PointCloudShuffle, ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader

MODEL_OBJ_DIR = './SileaneBunny.obj'


BATCH_SIZE = 1
NUM_POINT = 16384

def read_pcd(file_name, to_mm=True):
    with open(file_name, 'r') as f:
        begin = False
        points = []
        for line in f.readlines():
            if begin:
                xyz = list(map(float, line.strip().split()))
                if to_mm:
                    xyz[:3] = [ 1000*t for t in xyz[:3] ]
                points.append(xyz[:3])
            if line.startswith('DATA'):
                begin = True
    return np.array(points)

def extract_vertexes_from_obj(file_name):
    with open(file_name, 'r') as f:
        vertexes = []
        for line in f.readlines():
            line = line.strip()
            if line.startswith('v'):
                words = line.split()[1:]
                xyz = [float(w) for w in words]
                vertexes.append(xyz)
        ori_model_pc = np.array(vertexes)
        center = ( np.max(ori_model_pc, axis=0) + np.min(ori_model_pc, axis=0) ) / 2.0
        ori_model_pc = ori_model_pc - center
    return ori_model_pc

model_pointcloud = extract_vertexes_from_obj(MODEL_OBJ_DIR)
model_pointcloud *= 1000.0


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
type_brick = ObjectType(type_name='brick', class_idx=0, symmetry_type='finite',
                lambda_p=[[0.00973728, 0.0, 0.0], [0.0, 0.00512363, 0.0], [-0.0, 0.0, 0.00341862]],
                G=[[[1,0,0], [0,1,0], [0,0,1]],[[-1,0,0], [0,-1,0], [0,0,1]]])
backbone_config = {
    'npoint_per_layer': [4096,1024,256,64],
    'radius_per_layer': [[10,20,30],[30,45,60],[60,80,120],[120,160,240]]
}

net = PPRNetTest1(type_bunny, 'pointnet2msg', backbone_config, True, None, False)
net.to(device)

checkpoint_path = '../../logs/IPABunny_test1_conf_msg/log1_CE_only_conf_scale3/checkpoint.tar'
net, _, _ = load_checkpoint(checkpoint_path, net)

# dataset
transforms = transforms.Compose(
    [
        PointCloudShuffle(NUM_POINT),
        ToTensor()
    ]
)
print('Loading test dataset')
DATASET_DIR = 'you-path/Fraunhofer_IPA_Bin-Picking_dataset/h5_dataset/bunnp/train/'
TEST_CYCLE_RANGE = [740,741]
TEST_SCENE_RANGE = [1, 151]
test_dataset = IPAPoseDataset(DATASET_DIR, TEST_CYCLE_RANGE, TEST_SCENE_RANGE, transforms=transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print('Test dataset loaded, test point cloud size:', test_dataset.dataset['data'].shape)
#-----------------------END-----------------------


def eval_one_epoch(loader):
    net.eval() #

    for batch_idx, batch_samples in enumerate(loader):
        input_point_ori = batch_samples['point_clouds'].numpy()[0]
        inputs = {
            'point_clouds': batch_samples['point_clouds'].to(device),
            'labels': None
        }

        # Forward pass
        with torch.no_grad():
            time_start = time.time()
            pred_results, _ = net(inputs)
            print("Forward time:", time.time()-time_start)

        pred_trans_val = pred_results[0][0].cpu().numpy()
        pred_mat_val = pred_results[1][0].cpu().numpy()
        pred_vis_val = pred_results[2][0].cpu().numpy()
        pred_conf_val = torch.softmax(pred_results[4][0], dim=1)
        pred_conf_val = pred_conf_val.cpu().numpy()

        picked_idx = pred_conf_val[:,1] > 0.2
        # picked_idx = pred_vis_val > 0.5
        # picked_idx = np.bitwise_and(pred_vis_val > 0.4, pred_conf_val[:,1] > 0.5)
        # print(pred_conf_val.shape)
        # print(picked_idx.shape)

        input_point = input_point_ori[picked_idx]
        pred_trans_val = pred_trans_val[picked_idx]
        pred_mat_val = pred_mat_val[picked_idx]
        print('picked shape',pred_trans_val.shape)

        # print('pred_trans_val', pred_trans_val.shape)
        # print('pred_mat_val', pred_mat_val.shape)
        # pred_trans_val = pred_trans_val[0]
        # pred_mat_val = pred_mat_val
        
        ms = MeanShift(bandwidth=10, bin_seeding=True, cluster_all=False, min_bin_freq=40)
        ms.fit(pred_trans_val)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_


        # # Number of clusters in labels, ignoring noise if present. 
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(n_clusters)


        color_cluster = [np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]) for i in range(n_clusters)]
        color_per_point = np.ones([pred_trans_val.shape[0], pred_trans_val.shape[1]]) * 255
        for idx in range(color_per_point.shape[0]):
            if labels[idx] != -1:
                color_per_point[idx, :] = color_cluster[labels[idx]]
        

        pred_trans_cluster = [[] for _ in range(n_clusters)]
        pred_mat_cluster = [[] for _ in range(n_clusters)]
        for idx in range(pred_trans_val.shape[0]):
            if labels[idx] != -1:
                pred_trans_cluster[labels[idx]].append(np.reshape(pred_trans_val[idx], [1, 3]))
                pred_mat_cluster[labels[idx]].append(np.reshape(pred_mat_val[idx], [1, 3, 3]))
        pred_trans_cluster = [np.concatenate(cluster, axis=0) for cluster in pred_trans_cluster]
        pred_mat_cluster = [np.concatenate(cluster, axis=0) for cluster in pred_mat_cluster]

        cluster_center_pred = [ np.mean(cluster, axis=0) for cluster in pred_trans_cluster]


        cluster_mat_pred = []
        for mat_cluster in pred_mat_cluster:
            # print(mat_cluster)
            # print(mat_cluster.shape)
            all_quat = np.zeros([mat_cluster.shape[0], 4])
            for idx in range(mat_cluster.shape[0]):
                all_quat[idx] = eulerangles.mat2quat(mat_cluster[idx])
            quat = eulerangles.average_quat(all_quat)
            # print(ea.shape)
                # print(ea.shape)
            cluster_mat_pred.append( eulerangles.quat2mat(quat) )


        all_model_point = np.zeros([model_pointcloud.shape[0]*n_clusters, 3])
        all_model_color = np.zeros([model_pointcloud.shape[0]*n_clusters, 3])
        for cluster_idx in range(n_clusters):
            begin_idx = cluster_idx * model_pointcloud.shape[0]
            end_idx = (cluster_idx+1) * model_pointcloud.shape[0]
            all_model_color[begin_idx:end_idx, :] = color_cluster[cluster_idx]
            all_model_point[begin_idx:end_idx, :] = np.dot(model_pointcloud, cluster_mat_pred[cluster_idx].T) + \
                                                    np.tile(np.reshape(cluster_center_pred[cluster_idx], [1, 3]), [model_pointcloud.shape[0], 1])


        show3d_balls.showpoints(input_point_ori, ballradius=5)
        show3d_balls.showpoints(pred_trans_val, c_gt=color_per_point, ballradius=5)
        show3d_balls.showpoints(input_point, c_gt=color_per_point, ballradius=5)
        show3d_balls.showpoints(all_model_point, c_gt=all_model_color, ballradius=5)
        

if __name__ == "__main__":
    eval_one_epoch(test_loader)
