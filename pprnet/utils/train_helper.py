""" 
Helper classes and functions for pytorch training.

Author: Zhikai Dong
"""

import os
import torch
import torch.nn as nn
from collections import OrderedDict

def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn

class BNMomentumScheduler(object):
    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )
        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_bn_momentum(self, epoch):
        if epoch is None:
            epoch = self.last_epoch
        bn_momentum = self.lmbd(epoch)
        return bn_momentum

    
class OptimizerLRScheduler(object):
    """
        Helper class for steping learning rate. Pytorch lr_scheduler behaves differently before and after version 1.10.
        So for compatibility of all pytorch version, we use this class instead.
        It is a very simple scheduler which sets all learing rate to a given value.
    """
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise RuntimeError(
                "Class '{}' is not a PyTorch torch.optim.Optimizer".format(
                    type(optimizer).__name__
                )
            )
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda
        self.step(last_epoch + 1)   # initialize lr, this will update self.last_epoch
        self.last_epoch = last_epoch    # so we need to reset self.last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_lambda(epoch)
        # for param_group, lr in zip(self.optimizer.param_groups, self.lr_lambda(epoch)):
        #     param_group['lr'] = lr

    def get_optimizer_lr(self):
        lrs = [ g['lr'] for g in self.optimizer.param_groups ]
        return lrs

    # def get_lr(self, epoch):
    #     if epoch is None:
    #         epoch = self.last_epoch
    #     lr = self.lr_lambda(epoch)
    #     return lr

class SimpleLogger():
    def __init__(self, log_dir, file_path):
        if os.path.exists(log_dir):
            # raise RuntimeError('Dir:%s alreadly exist! Check if you really want to overwrite it.' % log_dir)
            pass
        else:
            os.makedirs(log_dir)
        os.system('cp %s %s' % (file_path, log_dir)) # bkp of train procedure
        self.log_file = open(os.path.join(log_dir, 'log_train.txt'), 'w')
        self.log_file.write('\n')
        self.cnt = 0
        self.state_dict = OrderedDict()

    def log_string(self, out_str):
        self.log_file.write(out_str+'\n')
        self.log_file.flush()
        print(out_str)

    def reset_state_dict(self, *args):
        self.cnt = 0
        self.state_dict = OrderedDict()
        for k in args:
            assert isinstance(k, str)
            self.state_dict[k] = 0.0
    
    def update_state_dict(self, state_dict):
        self.cnt += 1
        assert set(state_dict.keys()) == set(self.state_dict.keys())
        for k in state_dict.keys():
            self.state_dict[k] += state_dict[k]

    def print_state_dict(self, log=True, one_line=True, line_len=None):
        log_fn = self.log_string if log==True else print
        out_str = ''
        for i, (k,v) in enumerate(self.state_dict.items()):
            out_str += '%s: %f' % (k, 1.0*v/self.cnt)
            if i != len(self.state_dict.keys()):
                if line_len is not None and (i+1)%line_len==0:
                    out_str += '\n'
                else:
                    out_str += '\t' if one_line else '\n'
        log_fn(out_str)
        
