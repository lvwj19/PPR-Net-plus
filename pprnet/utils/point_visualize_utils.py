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
        assert len(point_array) == len(color_array)
        all_color = [ np.zeros( [ pnts.shape[0] ,3] ) for pnts in point_array]
        
        for i, c in enumerate(color_array):
            c=c.tolist()[0]
            all_color[i][:] = [c[1],c[0],c[2]]
        all_color = np.concatenate(all_color, axis=0)
    all_points = np.concatenate(point_array, axis=0)
    show3d_balls.showpoints(all_points, c_gt=all_color, ballradius=radius)
        
