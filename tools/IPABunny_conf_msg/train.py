import os
import sys
FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.dirname(FILE_PATH)
PROJECT_NAME = os.path.basename(FILE_DIR)
ROOT_DIR = os.path.dirname(os.path.dirname(FILE_DIR))
sys.path.append(ROOT_DIR)
import math
from datetime import datetime
import h5py
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
# from pprnet.pprnet import PPRNet, load_checkpoint, save_checkpoint
from pprnet.pprnet_plus import PPRNetPlus, load_checkpoint, save_checkpoint
from pprnet.object_type import ObjectType
from pprnet.utils.train_helper import BNMomentumScheduler, OptimizerLRScheduler, SimpleLogger
from pprnet.data.IPA_pose_dataset import IPAPoseDataset
from pprnet.data.pointcloud_transforms import PointCloudShuffle, ToTensor

#-----------------------1. GLOBAL SETTINGS START-----------------------
BATCH_SIZE = 8
NUM_POINT = 16384
MAX_EPOCH = 250
# log setting
LOG_NAME = 'log0_CE_scale3'
CHECKPOINT_PATH = None  # str or None, if is not None, model will continue to train from CHECKPOINT_PATH
# lr decay
BASE_LEARNING_RATE = 0.001
LR_DECAY_RATE = 0.7
MIN_LR = 1e-6 
LR_DECAY_STEP = 80
LR_LAMBDA = lambda epoch: max(BASE_LEARNING_RATE * LR_DECAY_RATE**(int(epoch / LR_DECAY_STEP)), MIN_LR)
# bn decay
# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MIN = 0.001
BN_DECAY_STEP = LR_DECAY_STEP
BN_DECAY_RATE = 0.5
BN_LAMBDA = lambda epoch: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(epoch / BN_DECAY_STEP)), BN_MOMENTUM_MIN)
# dataset settings
DATASET_DIR = 'your-path/Fraunhofer_IPA_Bin-Picking_dataset/h5_dataset/bunny/train/'
TRAIN_DATA_HOLD_EPOCH = 2
TRAIN_CYCLE_RANGES = [ [0,60], [60,120], [120,180], [180,240], [240,300], [300,360], [360,420], [420,480] ]
TRAIN_SCENE_RANGE = [1, 81]
TEST_CYCLE_RANGE = [480,490]
TEST_SCENE_RANGE = TRAIN_SCENE_RANGE
# train setting
EVAL_STAP = 1 # run evaluation after training EVAL_STAP epoch(s)
DISPLAY_BATCH_STEP = 100
#-----------------------GLOBAL SETTINGS END-----------------------

#-----------------------2. BUILD NETWORK AND OTHERS FROM GLOBAL SETTINGS-----------------------
# build logger
log_dir = os.path.join(ROOT_DIR, 'logs', PROJECT_NAME, LOG_NAME)
logger = SimpleLogger(log_dir, FILE_PATH)

# build net and optimizer( and load if CHECKPOINT_PATH is not None )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
type_bunny = ObjectType(type_name='bunny', class_idx=0, symmetry_type='finite',
                 lambda_p=[[0.0263663, 0.0, 0.0], [0.0, 0.0338224, 0.0], [-0.0, 0.0, 0.0484393]],
                 G=[ [[1,0,0], [0,1,0], [0,0,1]] ])
backbone_config = {
    'npoint_per_layer': [4096,1024,256,64],
    'radius_per_layer': [[20,30,40],[40,60,80],[80,120,160],[200,240,280]]
}
loss_weights = {
    'trans_head': 200.0/1000.0, # implicit convert mm to m by deviding 1000
    'rot_head': 200.0,
    'vis_head': 50.0,
    'conf_head': 10.0
}
net = PPRNetTest1(type_bunny, 'pointnet2msg', backbone_config, True, loss_weights, True)
if torch.cuda.device_count() > 1:
    logger.log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE)
if CHECKPOINT_PATH is not None:
    net, optimizer, start_epoch = load_checkpoint(CHECKPOINT_PATH, net, optimizer)
else:
    start_epoch = 0

# build scheduler
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=BN_LAMBDA, last_epoch=start_epoch-1)
lr_scheduler = OptimizerLRScheduler(optimizer, lr_lambda=LR_LAMBDA, last_epoch=start_epoch-1)

# dataset
transforms = transforms.Compose(
    [
        PointCloudShuffle(NUM_POINT),
        ToTensor()
    ]
)
print('Loading test dataset')
# train_dataset = IPAPoseDataset(DATASET_DIR, TRAIN_CYCLE_RANGES[0], TRAIN_SCENE_RANGE, transforms=transforms)
test_dataset = IPAPoseDataset(DATASET_DIR, TEST_CYCLE_RANGE, TRAIN_SCENE_RANGE, transforms=transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print('Test dataset loaded, test point cloud size:', test_dataset.dataset['data'].shape)
train_dataset = None
#-----------------------END-----------------------

def train_one_epoch(loader):
    logger.log_string('----------------TRAIN STATUS---------------')
    net.train() # set model to training mode
    
    logger.reset_state_dict('trans loss', 'rot loss', 'vs loss', 'dist', 'conf loss', 'conf acc', 'fg ratio')

    for batch_idx, batch_samples in enumerate(loader):
        labels = {
            'rot_label':batch_samples['rot_label'].to(device),
            'trans_label':batch_samples['trans_label'].to(device),
            'vis_label':batch_samples['vis_label'].to(device)
        }
        inputs = {
            'point_clouds': batch_samples['point_clouds'].to(device),
            'labels': labels
        }

        # Forward pass
        optimizer.zero_grad()
        pred_results, losses = net(inputs)
        losses['total'].backward()
        optimizer.step()

        dist_mean = torch.mean(torch.norm(pred_results[0].view(-1,3)-labels['trans_label'].view(-1,3), dim=1)).item()

        log_state_dict = {'trans loss':losses['trans_head'].item(), 'rot loss':losses['rot_head'].item(),
                            'vs loss':losses['vis_head'].item(), 'dist':dist_mean,
                            'conf loss':losses['conf_head'].item(), 'conf acc':losses['conf_accuracy'].item(), 
                            'fg ratio':losses['fg_ratio'].item()}
        logger.update_state_dict(log_state_dict)

        if batch_idx % DISPLAY_BATCH_STEP == 0 and batch_idx!= 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,len(loader)))
            logger.print_state_dict(log=False, line_len=4)

    print('Current batch/total batch num: %d/%d'%(len(loader),len(loader)))
    logger.print_state_dict(log=True, line_len=4)


def eval_one_epoch(loader):
    logger.log_string('----------------EVAL STATUS---------------')
    net.eval() # set model to eval mode

    logger.reset_state_dict('trans loss', 'rot loss', 'vs loss', 'dist', 'conf loss', 'conf acc', 'fg ratio')

    loss_sum = 0

    for batch_idx, batch_samples in enumerate(loader):
        labels = {
            'rot_label':batch_samples['rot_label'].to(device),
            'trans_label':batch_samples['trans_label'].to(device),
            'vis_label':batch_samples['vis_label'].to(device)
        }
        inputs = {
            'point_clouds': batch_samples['point_clouds'].to(device),
            'labels': labels
        }

        # Forward pass
        with torch.no_grad():
            pred_results, losses = net(inputs)

            loss_sum += losses['total'].item()
            dist_mean = torch.mean(torch.norm(pred_results[0].view(-1,3)-labels['trans_label'].view(-1,3), dim=1)).item()

            log_state_dict = {'trans loss':losses['trans_head'].item(), 'rot loss':losses['rot_head'].item(),
                                'vs loss':losses['vis_head'].item(), 'dist':dist_mean,
                                'conf loss':losses['conf_head'].item(), 'conf acc':losses['conf_accuracy'].item(), 
                                'fg ratio':losses['fg_ratio'].item()}
            logger.update_state_dict(log_state_dict)

    logger.print_state_dict(log=True, line_len=4)
    return loss_sum

def train(start_epoch):
    global train_dataset
    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        # loading dataset
        if epoch%TRAIN_DATA_HOLD_EPOCH == 0 or train_dataset is None:
            cid = int(epoch/TRAIN_DATA_HOLD_EPOCH) % len(TRAIN_CYCLE_RANGES)
            print('Loading train dataset...')
            train_dataset = IPAPoseDataset(DATASET_DIR, TRAIN_CYCLE_RANGES[cid], TRAIN_SCENE_RANGE, transforms=transforms)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
            print('Train dataset loaded, test point cloud size:', train_dataset.dataset['data'].shape)

        logger.log_string('************** EPOCH %03d **************' % (epoch))
        logger.log_string(str(datetime.now()))

        bnm_scheduler.step(epoch) # decay BN momentum
        lr_scheduler.step(epoch)

        logger.log_string('Current learning rate: %s'%str(lr_scheduler.get_optimizer_lr()))
        logger.log_string('Current BN decay momentum: %f'%(bnm_scheduler.get_bn_momentum(epoch)))

        train_one_epoch(train_loader)
        if epoch%EVAL_STAP == 0:
            loss = eval_one_epoch(test_loader)
            if loss < min_loss:
                min_loss = loss
                save_checkpoint(os.path.join(log_dir, 'checkpoint.tar'), epoch, net, optimizer, loss)
                logger.log_string("Model saved in file: %s" % os.path.join(log_dir, 'checkpoint.tar'))

if __name__=='__main__':
    train(start_epoch)

