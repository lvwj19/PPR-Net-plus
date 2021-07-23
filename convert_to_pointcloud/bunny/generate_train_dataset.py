import os
from H5DataGenerator import *

# output dirs
OUT_ROOT_DIR = '../../h5_dataset/bunny'
if not os.path.exists( OUT_ROOT_DIR ):
    os.mkdir(OUT_ROOT_DIR)
TRAIN_SET_DIR = os.path.join(OUT_ROOT_DIR, 'train')
if not os.path.exists( TRAIN_SET_DIR ):
    os.mkdir(TRAIN_SET_DIR)

# input dirs
IN_ROOT_DIR = '../../data_simulation/training/SileaneBunny_part_2'
GT_DIR = os.path.join(IN_ROOT_DIR, 'gt')
SEGMENT_DIR = os.path.join(IN_ROOT_DIR, 'p_segmentation')
DEPTH_DIR = os.path.join(IN_ROOT_DIR, 'p_depth')


if __name__ == "__main__":
    cycle_idx_list = range(250, 500)
    scene_idx_list = range(1, 81)
    g = H5DataGenerator(os.path.join(IN_ROOT_DIR, 'parameter.json'))
    for cycle_id in cycle_idx_list:
        # load background image
        bg_depth_image_path = os.path.join(DEPTH_DIR, 'cycle_{:0>4}'.format(cycle_id), '000_depth_uint16.png')
        bg_depth_image = cv2.imread(bg_depth_image_path,-1)

        out_cycle_dir = os.path.join(TRAIN_SET_DIR, 'cycle_{:0>4}'.format(cycle_id))
        if not os.path.exists(out_cycle_dir):
            os.mkdir(out_cycle_dir)
        for scene_id in scene_idx_list:
            # load inputs
            depth_image_path = os.path.join(DEPTH_DIR, 'cycle_{:0>4}'.format(cycle_id), '{:0>3}_depth_uint16.png'.format(scene_id))
            depth_image = cv2.imread(depth_image_path,-1)
            seg_img_path = os.path.join(SEGMENT_DIR, 'cycle_{:0>4}'.format(cycle_id), '{:0>3}_segmentation.png'.format(scene_id))
            segment_image = cv2.imread(seg_img_path,-1)
            gt_file_path = os.path.join(GT_DIR, 'cycle_{:0>4}'.format(cycle_id), '{:0>3}.csv'.format(scene_id))
            output_h5_path = os.path.join(out_cycle_dir, '{:0>3}.h5'.format(scene_id))

            g.process_train_set(depth_image, bg_depth_image, segment_image, gt_file_path, output_h5_path)

