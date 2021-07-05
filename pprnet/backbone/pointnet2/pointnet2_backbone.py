import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)
import torch
import torch.nn as nn
import pytorch_utils as pt_utils
from collections import namedtuple
from pointnet2_modules import PointnetFPModule, PointnetSAModule, PointnetSAModuleMSG

class Pointnet2Backbone(nn.Module):
    r"""
        PointNet2 backbone for pointwise feature extraction( single-scale grouping).

        Parameters
        ----------
        npoint_per_layer: List[int], length is 4
            number of sampled points per layer
        radius_per_layer: List[float], length is 4
            grouping radius per layer
        input_feature_dims: int = 0 
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, npoint_per_layer, radius_per_layer, input_feature_dims=0, use_xyz=True):
        super(Pointnet2Backbone, self).__init__()
        assert len(npoint_per_layer) == len(radius_per_layer) == 4

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=npoint_per_layer[0],
                radius=radius_per_layer[0],
                nsample=32,
                mlp=[input_feature_dims, 32, 32, 64],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=npoint_per_layer[1],
                radius=radius_per_layer[1],
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=npoint_per_layer[2],
                radius=radius_per_layer[2],
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=npoint_per_layer[3],
                radius=radius_per_layer[3],
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=use_xyz,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + input_feature_dims, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: torch.cuda.FloatTensor
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns:
            ----------
            new_features : torch.Tensor
                (B, 128, N) tensor. Pointwise feature.
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0]

class Pointnet2MSGBackbone(nn.Module):
    r"""
        PointNet2 backbone for pointwise feature extraction( multi-scale grouping).

        Parameters
        ----------
    """

    def __init__(self, npoint_per_layer, radius_per_layer, input_feature_dims=0, use_xyz=True):
        super(Pointnet2MSGBackbone, self).__init__()
        assert len(npoint_per_layer) == len(radius_per_layer) == 4
        self.nscale = len(radius_per_layer[0])

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=npoint_per_layer[0],
                radii=radius_per_layer[0],
                nsamples=[32]*self.nscale,
                mlps=[ [input_feature_dims, 32, 32, 64] for _ in range(self.nscale) ],
                use_xyz=use_xyz,
            )
        )
        c_out_0 = 64*self.nscale
        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=npoint_per_layer[1],
                radii=radius_per_layer[1],
                nsamples=[32]*self.nscale,
                mlps=[ [c_in, 64, 64, 128] for _ in range(self.nscale) ] ,
                use_xyz=use_xyz,
            )
        )
        c_out_1 = 128*self.nscale
        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=npoint_per_layer[2],
                radii=radius_per_layer[2],
                nsamples=[32]*self.nscale,
                mlps=[ [c_in, 128, 128, 256] for _ in range(self.nscale) ] ,
                use_xyz=use_xyz,
            )
        )
        c_out_2 = 256*self.nscale
        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=npoint_per_layer[3],
                radii=radius_per_layer[3],
                nsamples=[32]*self.nscale,
                mlps=[ [c_in, 256, 256, 512] for _ in range(self.nscale) ] ,
                use_xyz=use_xyz,
            )
        )
        c_out_3 = 512*self.nscale

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + input_feature_dims, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + c_out_0, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: torch.cuda.FloatTensor
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns:
            ----------
            new_features : torch.Tensor
                (B, 128, N) tensor. Pointwise feature.
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0]

if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = torch.randn(8, 16384, 3).cuda()
    xyz.requires_grad=True

    test_module = Pointnet2Backbone([4096,1024,256,64], [30,60,120,240])
    test_module.cuda()
    # print(test_module(xyz))


    for _ in range(1):
        new_features = test_module(xyz)
        print('new_features', new_features.shape)
        new_features.backward(torch.cuda.FloatTensor(*new_features.size()).fill_(1))
        # print(new_features)
        print(xyz.grad)
