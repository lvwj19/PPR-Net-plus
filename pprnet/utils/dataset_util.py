import os
import numpy as np
import random
import h5py
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, '..'))
import eulerangles

def initialize_dataset(data_path, split_ratio):
    file_names = os.listdir(data_path)
    random.shuffle(file_names)
    num_file = len(file_names)
    train_file = open(os.path.join(data_path, 'train_file_list.txt'), 'w')
    test_file = open(os.path.join(data_path, 'test_file_list.txt'), 'w')
    for i in range(num_file):
        if i < num_file*split_ratio:
            train_file.write(file_names[i]+'\n')
        else:
            test_file.write(file_names[i]+'\n')
    train_file.close()
    test_file.close()
    print('Dataset initialized')

def load_dataset(data_path, load_ratio=1.0, load_train_set=True, load_test_set=True):
    train_dataset = {'data':None, 'trans_label':None, 'rot_label':None, 'cls_label':None}
    test_dataset = {'data':None, 'trans_label':None, 'rot_label':None, 'cls_label':None}

    def _load(data_path, mode, load_ratio=1.0):
        data_list, trans_label_list, rot_label_list, cls_label_list = [], [], [], []
        num_point_in_h5 = 16384
        if mode =='train':
            split = 'train_file_list.txt'
        if mode =='test':
            split = 'test_file_list.txt'
        with open(os.path.join(data_path,split),'r') as train_file:
            train_names = train_file.readlines()
            for i,fn in enumerate(train_names):
                if i > len(train_names)*load_ratio:
                    break
                fn = fn.strip()
                if fn == '':
                    continue
                h5_filename = os.path.join(data_path, fn)
                f = h5py.File(h5_filename)
                data_list.append(f['data'][:].reshape(1, num_point_in_h5, 3))
                label = f['labels'][:]
                trans_label_list.append(label[:,:3].reshape(1, num_point_in_h5, 3))
                rot_quat = label[:,3:7]
                rot_mat = np.zeros([1,num_point_in_h5,3,3])
                for i in range(num_point_in_h5):
                    rot_mat[0,i] = eulerangles.quat2mat(rot_quat[i])
                rot_label_list.append(rot_mat)
                cls_label_list.append(label[:,8].reshape(1, num_point_in_h5))
            dataset={ 'data': np.concatenate(data_list, axis=0),  # shape: #scene,#point,3
                      'trans_label': np.concatenate(trans_label_list, axis=0),  # shape: #scene,#point,3
                      'rot_label': np.concatenate(rot_label_list, axis=0), # shape: #scene,#point,3，3
                      'cls_label': np.concatenate(cls_label_list, axis=0) # shape: #scene,#point
                    }
        return dataset

    if load_train_set:
        train_dataset = _load(data_path, 'train', load_ratio)
        print('train data loaded')
    if load_test_set:
        test_dataset = _load(data_path, 'test', load_ratio)
        print('test data loaded')
    return train_dataset, test_dataset

def load_dataset_by_cycle(data_path, cycle_idx_list, scene_idx_list, mode='train', collect_names=False, collect_error_names=False):
    num_point_in_h5 = 16384
    if mode =='train':
        data_list, trans_label_list, rot_label_list, vs_label_list, cls_label_list= [], [], [], [],[]
    else:
        data_list = []
    if collect_names:
        name_list = []
    if collect_error_names:
        error_name_list = []
    
    for cycle_id in cycle_idx_list:
        # print('Loading cycle: %d'%cycle_id)
        for scene_id in scene_idx_list:
            try:
                h5_file_name = os.path.join(data_path, 'cycle_{:0>4}'.format(cycle_id), '{:0>3}.h5'.format(scene_id))
                f = h5py.File(h5_file_name)
                data_list.append(f['data'][:].reshape(1, num_point_in_h5, 3))
                if mode == 'train':
                    label = f['labels'][:]
                    trans_label_list.append(label[:,:3].reshape(1, num_point_in_h5, 3))
                    rot_mat = label[:,3:12].reshape(1, num_point_in_h5, 3, 3)
                    rot_label_list.append(rot_mat)
                    vs = label[:, 12].reshape(1, num_point_in_h5)
                    vs_label_list.append(vs)
                    cls=label[:,-1].reshape(1, num_point_in_h5)
                    cls_label_list.append(cls)
                if collect_names:
                    name = 'cycle_{:0>4}'.format(cycle_id) + '_scene_{:0>3}'.format(scene_id)
                    name_list.append(name)
            except:
                print('Cycle %d scene %d error, please check' % (cycle_id, scene_id))
                if collect_error_names:
                    error_name = 'cycle_{:0>4}'.format(cycle_id) + '_scene_{:0>3}'.format(scene_id)
                    error_name_list.append(error_name)
                continue


    if mode == 'train':
        dataset={ 'data': np.concatenate(data_list, axis=0),  # shape: #scene,#point,3
                'trans_label': np.concatenate(trans_label_list, axis=0),  # shape: #scene,#point,3
                'rot_label': np.concatenate(rot_label_list, axis=0), # shape: #scene,#point,3�?
                'vs_label': np.concatenate(vs_label_list, axis=0),  # shape: #scene,#point
                'cls_label':np.concatenate(cls_label_list, axis=0),  # shape: #scene,#point
            }
    else:
        dataset={ 'data': np.concatenate(data_list, axis=0),  # shape: #scene,#point,3
            }
    if collect_names:
        dataset['name'] = name_list
    if collect_error_names:
        dataset['error_name'] = error_name_list
    return dataset

if __name__ == "__main__":
    # data_path = '/home/idesignlab/Desktop/synthetic_data_generation/cylinder_short_dataset/dataset_result'
    # initialize_dataset(data_path, 0.8)
    data_path = '/home/idesignlab/dongzhikai/Fraunhofer_IPA_Bin-Picking_dataset/h5_dataset/bunny/test'
    dataset = load_dataset_by_cycle(data_path, range(20, 21), range(1, 81), 'test')
    print('loaded')
