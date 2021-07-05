import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)
from pointnet2.pointnet2_backbone import Pointnet2Backbone
from pointnet2.pointnet2_backbone import Pointnet2MSGBackbone

#from rscnn.rscnn_backbone import RSCNNBackbone
