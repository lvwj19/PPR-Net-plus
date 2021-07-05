import os
import numpy as np
import random
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, '..'))
import eulerangles

import sklearn
from sklearn.cluster import MeanShift

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

def cluster_and_average(input_points,pred_vs, pred_trans, pred_mat, meanshift_kwargs):
    # ms = MeanShift(bandwidth=10, bin_seeding=True, cluster_all=False, min_bin_freq=100)
    ms = MeanShift(**meanshift_kwargs)
    ms.fit(pred_trans)
    labels = ms.labels_
    # cluster_centers = ms.cluster_centers_

    # # Number of clusters in labels, ignoring noise if present. 
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    pred_trans_cluster = [[] for _ in range(n_clusters)]
    pred_mat_cluster = [[] for _ in range(n_clusters)]
    pred_vs_cluster = [[] for _ in range(n_clusters)]
    for idx in range(input_points.shape[0]):
        if labels[idx] != -1:
            pred_trans_cluster[labels[idx]].append(np.reshape(pred_trans[idx], [1, 3]))
            pred_mat_cluster[labels[idx]].append(np.reshape(pred_mat[idx], [1, 3, 3]))
            pred_vs_cluster[labels[idx]].append(pred_vs[idx])
    pred_trans_cluster = [np.concatenate(cluster, axis=0) for cluster in pred_trans_cluster]
    pred_mat_cluster = [np.concatenate(cluster, axis=0) for cluster in pred_mat_cluster]
    pred_vs_cluster = [ np.mean(l) for l in pred_vs_cluster] 

    cluster_center_pred = [ np.mean(cluster, axis=0) for cluster in pred_trans_cluster]
    #vs_threshold=0.45
    #cluster_center_pred=cluster_center_pred[vs_cls_k>vs_threshold]
    #pred_mat_cluster=pred_mat_cluster[vs_cls_k>vs_threshold]
    cluster_mat_pred = []   
    for mat_cluster in pred_mat_cluster:
        all_quat = np.zeros([mat_cluster.shape[0], 4])
        for idx in range(mat_cluster.shape[0]):
            all_quat[idx] = eulerangles.mat2quat(mat_cluster[idx])
        quat = eulerangles.average_quat(all_quat)
        cluster_mat_pred.append( eulerangles.quat2mat(quat) )

    pc_segments = []
    for k in range(n_clusters):
        ii = np.where(labels==k)[0]
        pc_tmp = input_points[ii]
        pc_segments.append(pc_tmp)

    return n_clusters, pred_vs_cluster, labels, cluster_center_pred, cluster_mat_pred, pc_segments
