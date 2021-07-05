""" 
Pose loss function for PPRNet.

Author: Zhikai Dong
"""

import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        # if input.dim()>2:
        #     input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        #     input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        #     input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

# IMPORTANT: each type should have different PoseLossCalculator()!!!!!
class PoseLossCalculator():
    r"""
        Pose loss for PPRNet.
        Init args:
            symmetry_type: str, 'revolution' or 'finite'
            lambda_p: scalar(revolution) / List[List[float]] (3, 3) (finite)
            G: None(revolution) / List[ List[List[float]] (3, 3) ], len(G)==K, objects with K equal poses(finite)
            retoreflection: bool(revolution) / None(finite)
            
    """

    l1_loss = nn.L1Loss(reduction='none')
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    focal_loss = FocalLoss(gamma=1, alpha=0.2, size_average=True)

    def __init__(self, symmetry_type, lambda_p, G=None, retoreflection=None):
        assert symmetry_type in ['revolution', 'finite']
        self.symmetry_type = symmetry_type
        self.retoreflection = retoreflection
        self.lambda_p = lambda_p
        self.G = G

        self.ez_m_3_1 = None    # should be size of (m,3,1), where m = #points of this type
        self.G_list_m_3_3 = None    # should be list of Tensor with size (m,3,3), where m = #points of this type
        self.lambda_p_m_3_3 = None  # should be size of (m,3,3), where m = #points of this type

    @classmethod
    def confidence_loss(cls, pred_conf, conf_label, loss_type='CrossEntropyLoss'):
        assert loss_type in ['CrossEntropyLoss', 'FocalLoss']
        if loss_type=='CrossEntropyLoss':
            loss = torch.mean(cls.ce_loss(pred_conf, conf_label))
        elif loss_type=='FocalLoss':
            loss = cls.focal_loss(pred_conf, conf_label)
        return loss

    @classmethod
    def classification_loss(cls, pred_logits, cls_label, weight):
        """ 
        Calculate weighted classification loss.
        Args:
            pred_logits: torch.Tensor (M, ntype) 
            cls_label: torch.Tensor (M,) 
            weight: torch.Tensor (M,)  
        Returns:
            l: scalar, weight loss sum of all M samples
            w: scalar, sum of weight
            *Note* weighted average loss is l/w
        """
        M = pred_logits.shape[0]
        loss = cls.ce_loss(pred_logits, cls_label) # (M,) 
        if weight is not None:
            loss = torch.mul(loss, weight)    # (M,) 
            l = torch.sum(loss)    # scalar
            w = torch.sum(weight)   # scalar
        else:
            l = torch.sum(loss)    # scalar
            w = 1.0*M 
        return l, w

    @classmethod
    def trans_loss(cls, pred_trans, trans_label, weight, return_pointwise_loss=False):
        """ 
        Calculate weighted translation loss.
        Args:
            pred_trans: torch.Tensor (M, 3) 
            trans_label: torch.Tensor (M, 3) 
            weight: torch.Tensor (M,)  
        Returns:
            l: scalar, weight loss sum of all M samples
            w: scalar, sum of weight
            *Note* weighted average loss is l/w
        """
        M = pred_trans.shape[0]
        x = cls.l1_loss(pred_trans, trans_label)   # (M, 3) 
        x = torch.sum(x, dim=1)  # (M,) 

        if weight is not None:
            x = torch.mul(x, weight)    # (M,) 
            l = torch.sum(x)    # scalar
            w = torch.sum(weight)   # scalar
        else:
            l = torch.sum(x)
            w = 1.0*M 
        if not return_pointwise_loss:
            return l, w
        else:
            return l, w, x
        
    @classmethod
    def visibility_loss(cls, pred_vis, vis_label):
        """ 
        Calculate visibility loss(simple average l1 loss).
        Args:
            pred_vis: torch.Tensor (M,) 
            vis_label: torch.Tensor (M,) 
        Returns:
            loss: scalar
        """
        loss = torch.mean( torch.abs(pred_vis - vis_label) )
        return loss

    def rot_loss(self, rot_matrix, rot_label, weight, return_pointwise_loss=False):
        """ 
        Calculate weighted rotation loss.
        Args:
            rot_matrix: torch.Tensor (M, 3, 3) 
            rot_label: torch.Tensor (M, 3, 3) 
            weight: torch.Tensor (M,)  
        Returns:
            l: scalar , weight loss sum of all M samples
            w: sum of weight
            *Note* weighted average loss is l/w
        """
        if self.symmetry_type == 'revolution':
            rtn = self._rot_loss_revolution(rot_matrix, rot_label, self.lambda_p, self.retoreflection, weight, return_pointwise_loss)
        elif self.symmetry_type == 'finite':
            rtn = self._rot_loss_finite(rot_matrix, rot_label, self.lambda_p, self.G, weight, return_pointwise_loss)
        return rtn

    def _rot_loss_revolution(self, rot_matrix, rot_label, lambda_p, retoreflection, weight, return_pointwise_loss=False):
        """ 
        Calculate weighted rotation loss for revolution objects, privite helper function.
        Args:
            rot_matrix: torch.Tensor (M, 3, 3) 
            rot_label: torch.Tensor (M, 3, 3) 
            lambda_p: scalar
            retoreflection: bool
            weight: torch.Tensor (M,)  
        Returns:
            l: scalar , weight loss sum of all M samples
            w: sum of weight
            *Note* weighted average loss is l/w
        """
        dtype, device = rot_matrix.dtype, rot_matrix.device
        M = rot_matrix.shape[0]
        if self.ez_m_3_1 is None or self.ez_m_3_1.shape[0] != M:
            self.ez_m_3_1 = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device, requires_grad=False).view(1,3,1) # (1, 3, 1)
            self.ez_m_3_1 = self.ez_m_3_1.repeat(M, 1, 1) # (M, 3, 1)
        ez = self.ez_m_3_1
        if retoreflection==False:
            loss = lambda_p * torch.abs(torch.matmul(rot_matrix, ez) - torch.matmul(rot_label, ez)) # (M, 3, 1)
            loss = loss.squeeze(2)  # (M, 3)
            loss = torch.sum(loss, dim=1)  # (M,) 
        else:
            loss1 = lambda_p * torch.abs(torch.matmul(rot_matrix, ez) - torch.matmul(rot_label, ez)) # (M, 3, 1)
            loss2 = lambda_p * torch.abs(torch.matmul(rot_matrix, ez) + torch.matmul(rot_label, ez)) # (M, 3, 1)
            loss = torch.cat([loss1, loss2], dim=-1)    # (M, 3, 2)
            loss = loss.transpose(1, 2) # (M, 2, 3)
            loss = torch.sum(loss, dim=-1)  # (M, 2)
            loss = torch.min(loss, dim=-1)[0]   #(M,)
        x = loss    #(M,)

        if weight is not None:
            loss = torch.mul(loss, weight)    # (M,) 
            l = torch.sum(loss)    # scalar
            w = torch.sum(weight)   # scalar
        else:
            l = torch.sum(loss)    # scalar
            w = 1.0*M 

        if not return_pointwise_loss:
            return l, w
        else:
            return l, w, x

    def _rot_loss_finite(self, rot_matrix, rot_label, lambda_p, G, weight=None, return_pointwise_loss=False):
        """ 
        Calculate weighted rotation loss for objects finite symmetry, privite helper function.
        Args:
            rot_matrix: torch.Tensor (M, 3, 3) 
            rot_label: torch.Tensor (M, 3, 3) 
            lambda_p: List[List[float]] (3, 3)
            G: List[ List[List[float]] (3, 3)  ], len(G)==K, objects with K equal poses
            weight: torch.Tensor (M,)  
        Returns:
            l: scalar , weight loss sum of all M samples
            w: sum of weight
            *Note* weighted average loss is l/w
        """
        dtype, device = rot_matrix.dtype, rot_matrix.device
        M = rot_matrix.shape[0]
        K = len(G)
        if self.G_list_m_3_3 is None or self.G_list_m_3_3[0].shape[0]!=M:
            G = [  torch.tensor(g, dtype=dtype, device=device, requires_grad=False) for g in G ]
            self.G_list_m_3_3 = [ g.unsqueeze(0).repeat(M,1,1) for g in G ] # list of (M, 3, 3)
        G = self.G_list_m_3_3
        if self.lambda_p_m_3_3 is None or self.lambda_p_m_3_3.shape[0]!=M:
            lambda_p = torch.tensor(lambda_p, dtype=dtype, device=device, requires_grad=False).view(1, 3, 3)    # (1, 3, 3)
            self.lambda_p_m_3_3 = lambda_p.repeat(M, 1, 1)    # (M, 3, 3)
        lambda_p = self.lambda_p_m_3_3  # (M, 3, 3)

        P = torch.matmul(rot_matrix, lambda_p)  # (M, 3, 3)
        P = torch.unsqueeze(P, -1)  # (M, 3, 3, 1)
        P = P.repeat(1,1,1,K)   # (M, 3, 3, K)

        L_list = []
        for i in range(K):
            l = torch.matmul(torch.matmul(rot_label, G[i]), lambda_p) # (M, 3, 3)
            L_list.append(torch.unsqueeze(l, -1)) # (M, 3, 3, 1)
        L = torch.cat(L_list, dim=-1) # (M, 3, 3, K)

        sub = torch.abs(P - L).permute([0,3,1,2]).contiguous() # (M, K, 3, 3)
        sub = sub.view(M,K,9) # (M, K, 9)
        dist = torch.sum(sub, dim=-1) # (M, K)
        min_dist = torch.min(dist, dim=-1)[0] # (M,)
        x = min_dist

        if weight is not None:
            min_dist = torch.mul(min_dist, weight)    # (M,) 
            l = torch.sum(min_dist)    # scalar
            w = torch.sum(weight)   # scalar
        else:
            l = torch.sum(min_dist)    # scalar
            w = 1.0*M   # scalar

        if not return_pointwise_loss:
            return l, w
        else:
            return l, w, x

if __name__ == "__main__":
    x = torch.ones(10, 3)
    y = torch.zeros(10, 3)
    w = torch.ones(10) * 0.1
    
    loss_calculator1 = PoseLossCalculator('revolution', 1, retoreflection=True)
    loss_calculator2 = PoseLossCalculator('revolution', 1, retoreflection=False)
    print('loss', loss_calculator1.trans_loss(x, y, w))

    w = w.cuda()
    rot_matrix = torch.zeros(10, 3, 3).cuda()
    # rot_matrix[:, 1, 0] = 1.0
    # rot_matrix[:, 0, 1] = -1.0
    # rot_matrix[:, 2, 2] = 1.0
    rot_matrix[:, 0, 0] = 1.0
    rot_matrix[:, 1, 1] = -1.0
    rot_matrix[:, 2, 2] = -1.0
    rot_label = torch.zeros(10, 3, 3).cuda()
    rot_label[:, 0, 0] = 1.0
    rot_label[:, 1, 1] = 1.0
    rot_label[:, 2, 2] = 1.0
    print('loss', loss_calculator1.rot_loss(rot_matrix, rot_label, w))
    print('loss', loss_calculator2.rot_loss(rot_matrix, rot_label, w))
    print('loss', loss_calculator2.rot_loss(rot_matrix, rot_label, w))


    pred = torch.from_numpy( np.ones([2,3,3], dtype=np.float32) ).cuda()
    label = torch.from_numpy( np.ones([2,3,3], dtype=np.float32) ).cuda()
    label[0,...]*=0.9
    label[1,...]*=-1.0
    lambda_p = [ [1,0,0], [0,1,0], [0,0,1]]
    G = []
    G.append( [ [1,0,0], [0,1,0], [0,0,1] ] )
    G.append( [ [-1,0,0], [0,-1,0], [0,0,-1] ] )
    loss_calculator3 = PoseLossCalculator('finite',  lambda_p, G=G)
    print('loss', loss_calculator3.rot_loss(pred, label))

