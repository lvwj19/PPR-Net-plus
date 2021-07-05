import os
import numpy as np
import random
import h5py
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, '..'))
import show3d_balls 

def show_points(point_array, color_array=None, radius=3):
    assert isinstance(point_array, list)
    all_color = None
    if color_array is not None:
        if color_array == 'random':
            color_array = [np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]) for i in range(len(point_array))]
        assert len(point_array) == len(color_array)
        all_color = [ np.zeros( [ pnts.shape[0] ,3] ) for pnts in point_array]
        
        for i, c in enumerate(color_array):
            c=c.tolist()
            all_color[i][:] = [c[1],c[0],c[2]]
        all_color = np.concatenate(all_color, axis=0)
    all_points = np.concatenate(point_array, axis=0)
    show3d_balls.showpoints(all_points, c_gt=all_color, ballradius=radius)

def show_models(model_pc, trans, rot_mat, cls_idx, color_array=None, radius=3):
    assert len(trans) == len(rot_mat) == len(cls_idx)
    all_points = []
    all_color = [] if color_array is not None else None
    if color_array == 'random':
        color_array = [ [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for i in range(len(cls_idx))]
    for i in range(len(cls_idx)):
        model_pc_transformed = np.dot(model_pc[cls_idx[i]], rot_mat[i].T) + \
                                            np.tile(np.reshape(trans[i], [1, 3]), [model_pc[cls_idx[i]].shape[0], 1])
        all_points.append(model_pc_transformed)
        colors_tmp = np.tile(np.array(color_array[i]).reshape(1,3).astype(np.float32), [model_pc_transformed.shape[0], 1])
        if all_color is not None:
            all_color.append(colors_tmp)
    
    all_points = np.concatenate(all_points, axis=0)
    if all_color is not None:
        all_color = np.concatenate(all_color, axis=0)
    show3d_balls.showpoints(all_points, c_gt=all_color, ballradius=radius)

def get_models_points(model_pc, trans, rot_mat, cls_idx):
    assert len(trans) == len(rot_mat) == len(cls_idx)
    all_points = []
    for i in range(len(cls_idx)):
        model_pc_transformed = np.dot(model_pc[cls_idx[i]], rot_mat[i].T) + \
                                            np.tile(np.reshape(trans[i], [1, 3]), [model_pc[cls_idx[i]].shape[0], 1])
        all_points.append(model_pc_transformed)
    all_points = np.concatenate(all_points, axis=0)
    return all_points

        
