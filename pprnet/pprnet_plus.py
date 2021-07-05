""" 
Pytorch version of PPRNetPlus.

Author: Zhikai Dong
"""
import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)
import torch
import torch.nn as nn
import numpy as np
import backbone
from object_type import ObjectType
from pose_loss import PoseLossCalculator

# list all supported backbones
BACKBONE_SUPPORTED = ['pointnet2', 'pointnet2msg', 'rscnn']

class PPRNetPlus(nn.Module):
    r"""
        PPRNetPlus.

        Parameters
        ----------
        object_types: ObjectType or list of ObjectType
        backbone_name: str, see BACKBONE_SUPPORTED
        backbone_config: dict
            if backbone_name == 'pointnet2':
                keys: {npoint_per_layer,radius_per_layer,input_feature_dims(optional),use_xyz(optional)}
            -------------------------------------------------
        use_vis_branch: bool
        loss_weights: dict
            keys: {trans_head, rot_head, vis_head(optional), cls_head(optional)}
        return_loss: bool
    """
    def __init__(self, object_types, backbone_name, backbone_config, use_vis_branch, loss_weights, return_loss):
        assert backbone_name in BACKBONE_SUPPORTED
        super().__init__()
        self._set_up_constants()
        # self.num_point = 16384
        self.object_types = [object_types] if not isinstance(object_types, list) else object_types
        self.loss_calculators = [ PoseLossCalculator(**t.get_properties()) for t in self.object_types ]
        self.num_type = len(self.object_types)
        assert self.num_type > 0
        self.use_vis_branch = use_vis_branch
        self.use_cls_branch = self.num_type > 1
        self.loss_weights = loss_weights
        if return_loss:
            assert 'trans_head' in self.loss_weights and 'rot_head' in self.loss_weights and 'conf_head' in self.loss_weights
        self.return_loss = return_loss
        

        # Network basic building blocks
        # 1.backbone
        if backbone_name == 'pointnet2':
            self.backbone = backbone.Pointnet2Backbone(**backbone_config)
        elif backbone_name == 'pointnet2msg':
            self.backbone = backbone.Pointnet2MSGBackbone(**backbone_config)
        elif backbone_name == 'rscnn':
            self.backbone = backbone.RSCNNBackbone()
        backbone_feature_dim = 128  # by default, extracted feature dim is 128

        # 2.sementic classificaion head, predict pointwise sementic class
        if self.use_cls_branch:
            if return_loss:
                assert 'cls_head' in self.loss_weights
            self.cls_head = self._build_head([backbone_feature_dim, 128, 128, self.num_type])
            backbone_feature_dim += self.num_type   # feature concat with sementic prediction 
        else:
            self.cls_head = None

        # 3.translation head, regress pointwise relative translation to instance centroid
        self.trans_head = self._build_head([backbone_feature_dim, 128, 128, 3])

        # 4.rotation head, regress pointwise rotation in euler angle
        self.rot_head = self._build_head([backbone_feature_dim, 128, 128, 3])   

        # 5.visibility head, regress pointwise visibility 
        if self.use_vis_branch:
            if return_loss:
                assert 'vis_head' in self.loss_weights
            self.vis_head = self._build_head([backbone_feature_dim, 64, 64, 1])  
        else:
            self.vis_head = None

        # 6. confidence head, regress pointwise confidence 
        self.min_trans_threshold = 0.005
        self.min_rot_threshold = 0.005
        conf_input_feature_dim = backbone_feature_dim + 3 + 3
        if self.use_cls_branch:
            conf_input_feature_dim += self.num_type
        elif self.use_vis_branch:
            conf_input_feature_dim += 1
        self.conf_head = self._build_head([conf_input_feature_dim, 64, 64, 2]) # confidence for bg(0)/fg(1)  

    def forward(self, inputs):
        """ 
        Forward pass of the network
        Args:
            inputs: dict 
                keys: {point_clouds, labels(calc loss only)}
                -------------------------------------------------
                point_clouds: torch.Tensor 
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
                labels: dict
                    keys: {trans_label, rot_label, vis_label(optional), cls_label(optional)}
                    -------------------------------------------------
                    trans_label: torch.Tensor (M, 3) 
                    rot_label: torch.Tensor (M, 3, 3) 
                    vis_label: torch.Tensor (M,) 
                    cls_label: torch.Tensor (M,)  
        Returns:
            outputs: dict 
                keys: {total, trans_head, rot_head, vis_head(optional), cls_head(optional)}
        """
        batch_size = inputs['point_clouds'].shape[0]
        num_point = inputs['point_clouds'].shape[1]

        input_points = inputs['point_clouds']   # (B, N, 3)
        features = self.backbone(input_points)  # (B, 128, N)

        if self.use_cls_branch:
            pred_cls_logits = self.cls_head(features)   # (B, num_type, N)
            features = torch.cat([features, pred_cls_logits.detach()], 1)   # (B, num_type+128, N)
            pred_cls_logits = pred_cls_logits.transpose(1, 2).contiguous()  # (B, N, num_type)
            pred_cls_logits_flatten = pred_cls_logits.view(batch_size*num_point, self.num_type) # (B*N, num_type)
        else:
            pred_cls_logits = None
            pred_cls_logits_flatten = None

        # translation prediction
        pred_offsets = self.trans_head(features).transpose(1, 2).contiguous()   # (B, N, 3)
        pred_centroids = input_points + pred_offsets    # (B, N, 3)
        pred_centroids_flatten = pred_centroids.view(batch_size*num_point, 3)   # (B*N, 3)

        # rotation prediction
        pred_euler_angles = self.rot_head(features).transpose(1, 2).contiguous()    # (B, N, 3)
        pred_mat_flatten = self._euler_angle_to_rotation_matrix(pred_euler_angles.view(batch_size*num_point, 3))   # (B*N, 3, 3)
        pred_mat = pred_mat_flatten.view(batch_size, num_point, 3, 3)   # (B, N, 3, 3)

        # visibility prediction
        if self.use_vis_branch:
            pred_visibility = self.vis_head(features).squeeze(1)  # (B, N)
            pred_visibility_flatten = pred_visibility.view(batch_size*num_point)  # (B*N,)
        else:
            pred_visibility = None
            pred_visibility_flatten = None

        # confidence prediction
        feats_to_concat = [features] # (B, num_type+128, N) or (B, 128, N)
        feats_to_concat.append(pred_centroids.detach().transpose(1, 2)) # (B, 3, N)
        feats_to_concat.append(pred_euler_angles.detach().transpose(1, 2)) # (B, 3, N)
        if self.use_vis_branch:
            feats_to_concat.append(pred_visibility.detach().unsqueeze(1)) # (B, 1, N)
        features_conf = torch.cat(feats_to_concat, 1)
        pred_conf = self.conf_head(features_conf).transpose(1, 2).contiguous()   # (B, N, 2)
        pred_conf_flatten = pred_conf.view(batch_size*num_point, 2) # (B*N, 2)


        pred_results = [ pred_centroids, pred_mat, pred_visibility, pred_cls_logits, pred_conf ]
        pred_results_flatten = [ pred_centroids_flatten, pred_mat_flatten, pred_visibility_flatten, pred_cls_logits_flatten, pred_conf_flatten ]

        if self.return_loss:   # calculate pose loss
            losses = self._compute_loss(pred_results_flatten, inputs['labels'])
        else:
            losses = None

        return pred_results, losses

    def _compute_loss(self, preds_flatten, labels):
        """ 
        Forward pass of the network
        Args:
            preds_flatten: list
                [ pred_centroids_flatten, pred_mat_flatten, pred_visibility_flatten, pred_cls_logits_flatten, pred_conf_flatten ]
            labels: dict
                keys: {trans_label, rot_label, vis_label(optional), cls_label(optional)}
                -------------------------------------------------
                trans_label: torch.Tensor (M, 3) 
                rot_label: torch.Tensor (M, 3, 3) 
                vis_label: torch.Tensor (M,) 
                cls_label: torch.Tensor (M,)  
        Returns:
            outputs: dict 
                keys: {total, trans_head, rot_head, vis_head(optional), cls_head(optional)}
        """
        pred_centroids_flatten, pred_mat_flatten, \
            pred_visibility_flatten, pred_cls_logits_flatten, pred_conf_flatten = preds_flatten
        batch_size, num_point = labels['trans_label'].shape[0:2]
        if self.use_vis_branch:
            vis_label_flatten = labels['vis_label'].view(batch_size*num_point)  # (B*N,)
        else:
            vis_label_flatten = None
        if self.use_cls_branch:
            cls_label_flatten = labels['cls_label'].view(batch_size*num_point)  # (B*N,)
        trans_label_flatten = labels['trans_label'].view(batch_size*num_point, 3)   # (B*N, 3)
        rot_label_flatten = labels['rot_label'].view(batch_size*num_point, 3, 3)    # (B*N, 3, 3)

        losses = dict()
        if self.num_type == 1:
            assert self.use_cls_branch == False
            if self.use_vis_branch:
                losses['vis_head'] = PoseLossCalculator.visibility_loss(pred_visibility_flatten, vis_label_flatten) \
                                                            * self.loss_weights['vis_head']
            # trans loss
            l, w, x = PoseLossCalculator.trans_loss(pred_centroids_flatten,
                                                        trans_label_flatten,  
                                                        vis_label_flatten,
                                                        return_pointwise_loss=True)
            x = x/1000.0    # convert to m
            losses['trans_x']=x
            trans_threshold = max(self.min_trans_threshold, torch.mean(x).item()/2)
            # trans_threshold = 0.01
            conf_label_trans = x < trans_threshold  # (M, )
            losses['trans_head'] = (l / w) * self.loss_weights['trans_head']
            # rot loss
            l, w, x = self.loss_calculators[0].rot_loss(pred_mat_flatten,
                                                        rot_label_flatten,  
                                                        vis_label_flatten,
                                                        return_pointwise_loss=True)
            rot_threshold = max(self.min_rot_threshold, torch.mean(x).item()/2)
            # rot_threshold = 0.01
            conf_label_rot = x < rot_threshold  # (M, )
            losses['rot_x'] =x
            losses['rot_head'] = (l / w) * self.loss_weights['rot_head']
            # confidence loss
            conf_label_flatten = (conf_label_trans & conf_label_rot).type(torch.int64)  # (M, )
            conf_loss = self.loss_calculators[0].confidence_loss(pred_conf_flatten,
                                                             conf_label_flatten, 
                                                             loss_type='CrossEntropyLoss')
            losses['conf_head'] = conf_loss * self.loss_weights['conf_head']
            # calculate accuracy of confidence branch and store in losses for evaluation
            losses['fg_ratio'] = torch.sum(conf_label_flatten).type(torch.float32) / conf_label_flatten.shape[0]
            correct = torch.argmax(pred_conf_flatten, dim=1) == conf_label_flatten  # (M, )
            acc = torch.sum(correct).type(torch.float32) / correct.shape[0]
            losses['conf_accuracy'] = acc
        else:
            assert self.use_cls_branch == True
            if self.use_vis_branch:
                losses['vis_head'] = PoseLossCalculator.visibility_loss(pred_visibility_flatten, 
                                                                                vis_label_flatten) * self.loss_weights['vis_head']
            l, w = PoseLossCalculator.classification_loss(pred_cls_logits_flatten, 
                                                                cls_label_flatten,
                                                                vis_label_flatten)       
            losses['cls_head'] = (l / w) * self.loss_weights['cls_head']
            l, w = PoseLossCalculator.trans_loss(pred_centroids_flatten, 
                                                        trans_label_flatten,  
                                                        vis_label_flatten)  
            losses['trans_head'] = (l / w) * self.loss_weights['trans_head']
            tmp_ls, tmp_ws = [], []
            cnt = 0 
            for obj_idx, obj_type in enumerate(self.object_types):
                picked_idx = torch.nonzero( cls_label_flatten==obj_type.class_idx ).view(-1)
                num_picked = picked_idx.shape[0]
                if num_picked == 0:
                    continue
                cnt += num_picked
                vis_label_flatten_picked = vis_label_flatten[picked_idx] if vis_label_flatten is not None else None
                l, w = self.loss_calculators[obj_idx].rot_loss(pred_mat_flatten[picked_idx],
                                                                rot_label_flatten[picked_idx],
                                                                vis_label_flatten_picked)
                tmp_ls.append(l)
                tmp_ws.append(w)
            assert cnt == batch_size*num_point
            losses['rot_head'] = (sum(tmp_ls) / sum(tmp_ws)) * self.loss_weights['rot_head']

        losses['total'] = losses['trans_head'] +  losses['rot_head'] + losses['conf_head']
        if self.use_cls_branch:
            losses['total'] += losses['cls_head']
        if self.use_vis_branch:
            losses['total'] += losses['vis_head']

        return losses

    def _build_head(self, nchannels):
        """ 
        Help function for building regresstion or classification head.
        Args:
            nchannels: List[int]
                input and output channels of each layer. 
                nchannels[0] and nchannels[1] are (in/out)put for layer 0,
                nchannels[1] and nchannels[2] are (in/out)put for layer 1...

        Returns:
            head: torch.nn.Sequential()
        """
        assert len(nchannels) > 1
        num_layers = len(nchannels) - 1

        head = nn.Sequential()
        for idx in range(num_layers):
            if idx != num_layers - 1:
                head.add_module( "conv_%d"%(idx+1), nn.Conv1d(nchannels[idx], nchannels[idx+1], 1))
                head.add_module( "bn_%d"%(idx+1), nn.BatchNorm1d(nchannels[idx+1]))
                head.add_module( "relu_%d"%(idx+1), nn.ReLU())
            else:   # last layer don't have bn and relu
                head.add_module( "conv_%d"%(idx+1), nn.Conv1d(nchannels[idx], nchannels[idx+1], 1))
        return head

    def _euler_angle_to_rotation_matrix(self, angles):
        """ 
        Convert euler angles to rotation matrix representations.
        Args:
            angles: torch.Tensor (M, 3) 
                2D tensor.
        Returns:
            rot_matrix: torch.Tensor (M, 3, 3) 
        """
        rot_x, rot_y, rot_z = angles[:, :1], angles[:, 1:2], angles[:, 2:3] #(M, 1)
        cos_rot_x, sin_rot_x = torch.cos(rot_x), torch.sin(rot_x) #(M, 1)
        cos_rot_y, sin_rot_y = torch.cos(rot_y), torch.sin(rot_y) #(M, 1)
        cos_rot_z, sin_rot_z = torch.cos(rot_z), torch.sin(rot_z) #(M, 1)
        if self.ones_m_1 is None or self.ones_m_1.shape[0]!=rot_x.shape[0]:
            self.ones_m_1 = torch.ones_like(cos_rot_x) #(M, 1)
            self.zeros_m_1 = torch.zeros_like(cos_rot_x) #(M, 1)
        one = self.ones_m_1
        zero = self.zeros_m_1

        rot_x = torch.stack([torch.cat([one, zero, zero], dim=1),
                        torch.cat([zero, cos_rot_x, sin_rot_x], dim=1),
                        torch.cat([zero, -sin_rot_x, cos_rot_x], dim=1)], dim=1)

        rot_y = torch.stack([torch.cat([cos_rot_y, zero, -sin_rot_y], dim=1),
                        torch.cat([zero, one, zero], dim=1),
                        torch.cat([sin_rot_y, zero, cos_rot_y], dim=1)], dim=1)

        rot_z = torch.stack([torch.cat([cos_rot_z, sin_rot_z, zero], dim=1),
                        torch.cat([-sin_rot_z, cos_rot_z, zero], dim=1),
                        torch.cat([zero, zero, one], dim=1)], dim=1)

        rot_matrix = torch.matmul(rot_x, torch.matmul(rot_y, rot_z))

        return rot_matrix

    def _set_up_constants(self):
        """ 
        Set up contants to avoid re-creating if not necessary.
        """
        self.ones_m_1 = None    # should be size of (m,1), where m = batch_size * num_point
        self.zeros_m_1 = None   # should be size of (m,1), where m = batch_size * num_point


# Helper function for saving and loading network
def load_checkpoint(checkpoint_path, net, optimizer=None, strict=True):
    """ 
    Load checkpoint for network and optimizer.
    Args:
        checkpoint_path: str
        net: torch.nn.Module
        optimizer(optional): torch.optim.Optimizer or None
    Returns:
        net: torch.nn.Module
        optimizer: torch.optim.Optimizer
        start_epoch: int
    """
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(checkpoint_path, start_epoch))
    return net, optimizer, start_epoch

def save_checkpoint(checkpoint_path, current_epoch, net, optimizer, loss):
    """ 
    Save checkpoint for network and optimizer.
    Args:
        checkpoint_path: str
        current_epoch: int, current epoch index
        net: torch.nn.Module
        optimizer: torch.optim.Optimizer or None
        loss:
    """
    save_dict = {'epoch': current_epoch+1, # after training one epoch, the start_epoch should be epoch+1
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }
    try: # with nn.DataParallel() the net is added as a submodule of DataParallel
        save_dict['model_state_dict'] = net.module.state_dict()
    except:
        save_dict['model_state_dict'] = net.state_dict()
    torch.save(save_dict, checkpoint_path)


# if __name__ == "__main__":
#     # import numpy as np
#     # xyz_np = np.ones([8,16384,3], dtype=np.float)
#     # xyz = torch.from_numpy(xyz_np).cuda()
#     # print(xyz)

#     backbone_config = {
#         'npoint_per_layer': [4096,1024,256,64],
#         'radius_per_layer': [30,60,120,240]
#     }
#     net = PPRNet(2, 'pointnet2', backbone_config, True)
#     # # print(net)
#     # # exit()
#     # net.cuda()

#     # test _euler_angle_to_rotation_matrix()
#     angle = torch.Tensor([[3.14159/2,0.0,0.0]])
#     angle = angle.cuda()
#     print(angle)
#     print(angle.size())
#     print(angle.shape[0])
#     # net = PPRNet(2, 'pointnet2', backbone_config, True)
#     mat = net._euler_angle_to_rotation_matrix(angle)
#     print(mat, mat.shape, mat.dtype, mat.device)
#     angle = torch.Tensor([[3.14159/2,0.0,0.0], [3.14159/2,0.0,0.0]])
#     angle = angle.cuda()
#     mat = net._euler_angle_to_rotation_matrix(angle)
#     print(mat, mat.shape, mat.dtype, mat.device)
#     mat = net._euler_angle_to_rotation_matrix(angle)
#     print(mat, mat.shape, mat.dtype, mat.device)
#     exit()




#     torch.manual_seed(1)
#     torch.cuda.manual_seed_all(1)
#     xyz = torch.randn(8, 16384, 3).cuda()
#     xyz.requires_grad=True



#     for _ in range(1):
#         output = net({'point_clouds':xyz})
#         for o in output:
#             print(o.shape) if o is not None else print(None)
#         # features.backward(torch.cuda.FloatTensor(*features.size()).fill_(1))
#         # # print(new_features)
#         # # print(xyz.grad)
#         # print('xyz', xyz.shape)

