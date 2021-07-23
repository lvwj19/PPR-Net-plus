import json
import math
import numpy as np
import cv2
import os
import tensorflow as tf
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
import h5py
import time

# from show3d_balls import showpoints

class H5DataGenerator(object):
    def __init__(self, params_file_name, target_num_point = 16384):
        '''
        Input:
            params_file_name: path of parameter file ("parameter.json")
            target_num_point: target number of sampled points, default is 16384
        '''
        self.params = self._load_parameters(params_file_name)
        self.target_num_point = target_num_point

        # for fps
        with tf.device('/gpu:0'):
            self.input_point_pl = tf.placeholder(tf.float32, shape=(1, None, 3))
            self.sampled_idx_op = farthest_point_sample(16384,self.input_point_pl)
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)

    def process_train_set(self, depth_img, bg_depth_img, segment_img, gt_file_path, output_file_path, xyz_limit=None, verbose=True):
        '''
        Input:
            depth_img: np array of depth image, dtype is uint16
            bg_depth_img: np array of background depth image, dtype is uint16
            segment_img: np array of segment image, dtype is uint8
            gt_file_path: str
            output_file_path: str, output h5 path
            xyz_limit: None if no limit for xyz. Typical [ [xmin, xmax], [ymin, ymax], [zmin, zmax] ]
            verbose: whether to display logging info
        '''
        if verbose:
            start_time = time.time()
        # step 1: check and parse input
        assert depth_img.shape == (self.params['resolutionY'], self.params['resolutionX']) and depth_img.dtype == np.uint16
        assert bg_depth_img.shape == depth_img.shape and bg_depth_img.dtype == np.uint16
        assert segment_img.shape == depth_img.shape and segment_img.dtype == np.uint8
        label_trans, label_rot, label_vs = self._read_label_csv(gt_file_path)

        # step 2: convet foregroud pixel to 3d points, and extract its object ids
        ys, xs = np.where(depth_img != bg_depth_img)
        zs = depth_img[ys, xs]
        obj_ids = segment_img[ys, xs]
        ys = ys + self.params['pixelOffset_Y_KoSyTopLeft']
        xs = xs + self.params['pixelOffset_X_KoSyTopLeft']
        points = self._depth_to_pointcloud_optimized(xs, ys, zs, to_mm=False, xyz_limit=xyz_limit)

        # step 3: sample or pad to target_num_point
        # if len(points) <= target_num_point, pad to target_num_point
        num_pnt = points.shape[0]
        if num_pnt == 0:
            print('No foreground points!!!!!')
            return
        if num_pnt <= self.target_num_point:
            t = int(1.0 * self.target_num_point / num_pnt) + 1
            points_tile = np.tile(points, [t, 1])
            points = points_tile[:self.target_num_point]
            obj_ids_tile = np.tile(obj_ids, [t])
            obj_ids = obj_ids_tile[:self.target_num_point]
        # if len(points) > target_num_point, using fps to sample to target_num_point
        if num_pnt > self.target_num_point:
            sampled_idx = self.sess.run(self.sampled_idx_op, feed_dict={self.input_point_pl: points.reshape([1, -1, 3])})
            sampled_idx = sampled_idx.reshape([-1])
            # showpoints(points, ballradius=3)
            points = points[sampled_idx]
            # showpoints(points, ballradius=3)
            obj_ids = obj_ids[sampled_idx]

        # step 4: collect labels
        obj_ids[np.where(obj_ids>len(label_trans))] = 0
        label_trans = label_trans[obj_ids]
        label_rot = label_rot[obj_ids]
        label_vs = label_vs[obj_ids]
        # reset background points translation label
        bg_ids = np.where(obj_ids==0)[0]
        num_bg_pnt = len(bg_ids)
        label_trans[bg_ids] = points[bg_ids]
        labels = np.concatenate( [label_trans, label_rot, label_vs.reshape([-1, 1]), obj_ids.reshape([-1, 1])], axis=-1 )
        assert points.shape == (self.target_num_point, 3) and labels.shape == (self.target_num_point, 14)

        # step 5: save as h5 file
        if not os.path.exists(output_file_path):
            with h5py.File(output_file_path) as f:
                f['data'] = points
                f['labels'] = labels
                if verbose:
                    t = time.time() - start_time
                    print('Successfully write to %s in %f seconds.' % (output_file_path, t))
                    print('Foreground point number: %d\t Background point number: %d' % (num_pnt, num_bg_pnt))
                    if num_pnt < self.target_num_point:
                        print('Waring: not enough points, padded to target number')
                    if num_bg_pnt > 0:
                        print('Waring: contains background points')

    def process_test_set(self, depth_img, bg_depth_img, output_file_path, xyz_limit=None, verbose=True):
        '''
        Input:
            depth_img: np array of depth image, dtype is uint16
            bg_depth_img: np array of background depth image, dtype is uint16
            output_file_path: str, output h5 path
            xyz_limit: None if no limit for xyz. Typical [ [xmin, xmax], [ymin, ymax], [zmin, zmax] ]
            verbose: whether to display logging info
        '''
        if verbose:
            start_time = time.time()
        # step 1: check and parse input
        assert depth_img.shape == (self.params['resolutionY'], self.params['resolutionX']) and depth_img.dtype == np.uint16
        assert bg_depth_img.shape == depth_img.shape and bg_depth_img.dtype == np.uint16

        # step 2: convet foregroud pixel to 3d points, and extract its object ids
        ys, xs = np.where(depth_img != bg_depth_img)
        zs = depth_img[ys, xs]
        ys = ys + self.params['pixelOffset_Y_KoSyTopLeft']
        xs = xs + self.params['pixelOffset_X_KoSyTopLeft']
        points = self._depth_to_pointcloud_optimized(xs, ys, zs, to_mm=False, xyz_limit=xyz_limit)

        # step 3: sample or pad to target_num_point
        # if len(points) <= target_num_point, pad to target_num_point
        num_pnt = points.shape[0]
        if num_pnt == 0:
            print('No foreground points!!!!!')
            return
        if num_pnt <= self.target_num_point:
            t = int(1.0 * self.target_num_point / num_pnt) + 1
            points_tile = np.tile(points, [t, 1])
            points = points_tile[:self.target_num_point]
        # if len(points) > target_num_point, using fps to sample to target_num_point
        if num_pnt > self.target_num_point:
            sampled_idx = self.sess.run(self.sampled_idx_op, feed_dict={self.input_point_pl: points.reshape([1, -1, 3])})
            sampled_idx = sampled_idx.reshape([-1])
            points = points[sampled_idx]

        # step 4: save as h5 file
        if not os.path.exists(output_file_path):
            with h5py.File(output_file_path) as f:
                f['data'] = points
                if verbose:
                    t = time.time() - start_time
                    print('Successfully write to %s in %f seconds.' % (output_file_path, t))
                    print('Foreground point number: %d' % num_pnt)
                    if num_pnt < self.target_num_point:
                        print('Waring: not enough points, padded to target number')

    def _depth_to_pointcloud_optimized(self, us, vs, zs, to_mm = False, xyz_limit=None):
        '''
        Input:
            us: np array of u coordinate
            vs: np array of v coordinate
            zs: np array of z coordinate
            to_mm: *1000.0 if True
            xyz_limit: None if no limit for xyz. Typical [ [xmin, xmax], [ymin, ymax], [zmin, zmax] ]
        '''
        assert len(us) == len(vs) == len(zs)
        camera_info = self.params
        fx = camera_info['fu']
        fy = camera_info['fv']
        cx = camera_info['cu']
        cy = camera_info['cv']
        clip_start = camera_info['clip_start']
        clip_end = camera_info['clip_end']

        Zcs = (clip_start + (zs/float(camera_info['max_val_in_depth'])) * (clip_end - clip_start))
        if to_mm:
            Zcs *= 1000
        Xcs = -(us - cx) * Zcs / fx
        Ycs = -(vs - cy) * Zcs / fy
        Xcs = np.reshape(Xcs, (-1, 1))
        Ycs = np.reshape(Ycs, (-1, 1))
        Zcs = np.reshape(Zcs, (-1, 1))
        points = np.concatenate([Xcs, Ycs, Zcs], axis=-1)

        if xyz_limit is not None:
            if xyz_limit[0] is not None:
                xmin, xmax = xyz_limit[0]
                if xmin is not None:
                    idx = np.where( points[:, 0]>xmin )
                    points = points[idx]
                if xmax is not None:
                    idx = np.where( points[:, 0]<xmax )
                    points = points[idx]
            if xyz_limit[1] is not None:
                ymin, ymax = xyz_limit[1]
                if ymin is not None:
                    idx = np.where( points[:, 1]>ymin )
                    points = points[idx]
                if ymax is not None:
                    idx = np.where( points[:, 1]<ymax )
                    points = points[idx]
            if xyz_limit[2] is not None:
                zmin, zmax = xyz_limit[2]
                if zmin is not None:
                    idx = np.where( points[:, 2]>zmin )
                    points = points[idx]
                if zmax is not None:
                    idx = np.where( points[:, 2]<zmax )
                    points = points[idx]
                
        return points

    def _load_parameters(self, params_file_name):
        '''
        Input:
            params_file_name: path of parameter file ("parameter.json")
        '''
        params = {}
        with open(params_file_name,'r') as f:
            config = json.load(f)
            params = config
            # compute fu and fv
            angle = params['perspectiveAngle'] * math.pi / 180.0
            f = 1.0 /  ( 2*math.tan(angle/2.0) ) * params['resolution_big']
            params['fu'] = f
            params['fv'] = f
            params['cu'] = params['resolution_big'] / 2.0
            params['cv'] = params['cu']
            params['max_val_in_depth'] = 65535.0
        return params 

    def _read_label_csv(self, file_name):
        '''
        Input:
            file_name: path of ground truth file name
        Output:
            label_trans: numpy array of shape (num_obj+1)*3. 0th pos is bg
            label_rot: numpy array of shape (num_obj+1)*9. 0th pos is bg
            label_vs: numpy array of shape (num_obj+1,). 0th pos is bg
        '''
        num_obj = int(os.path.basename(file_name).split('.')[0])
        label_trans, label_rot, label_vs = [], [], []
        with open(file_name, 'r') as f:
            for line in f.readlines()[1:]:
                line = line.strip()
                if len(line) == 0:
                    continue
                words = line.split(',')
                id = int(words[0])
                if id == 0:
                    label_trans.append([0, 0, 0])
                    label_rot.append( np.eye(3).reshape(-1) )
                    label_vs.append(0.0)
                if id > 0:
                    label_trans.append( list(map(float, words[2:5])) )
                    R = np.array(list(map(float, words[5:14]))).reshape((3,3)).T.reshape(-1)
                    label_rot.append( R )
                    label_vs.append( float(words[-1]) )

        label_trans = np.array(label_trans)
        label_rot = np.array(label_rot)
        label_vs = np.array(label_vs)
        assert label_trans.shape == (num_obj+1,3) and label_rot.shape == (num_obj+1,9) and label_vs.shape == (num_obj+1,) 
        return label_trans, label_rot, label_vs
