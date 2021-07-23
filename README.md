## PPR-Net++: Accurate 6D Pose Estimation in Stacked Scenarios
This is the code of pytorch version for our IROS2019 paper and another journal paper: **PPR-Net: point-wise pose regression network for instance segmentation and 6d pose estimation in bin-picking scenarios; PPR-Net++: Accurate 6D Pose Estimation in Stacked Scenarios**.


## Environment
Ubuntu 16.04/18.04

python3.6, torch 1.1.0, torchvision 0.3.0, opencv-python, sklearn, h5py, nibabel, et al.

## Dataset
Siléane dataset is available at [here](http://rbregier.github.io/dataset2017).

Fraunhofer IPA Bin-Picking dataset is available at [here](https://owncloud.fraunhofer.de/index.php/s/AacICuOWQVWDDfP?path=%2F).

## Evaluation metric
The python code of evaluation metric is available at [here](https://github.com/rbregier/pose_recovery_evaluation).

## Citation
If you use this codebase in your research, please cite:
```
@inproceedings{pprnet19IROS,
  title={PPR-Net: point-wise pose regression network for instance segmentation and 6d pose estimation in bin-picking scenarios},
  author={Dong, Zhikai and Liu, Sicheng and Zhou, Tao and Cheng, Hui and Zeng, Long and Yu, Xingyao and Liu, Houde},
  booktitle={2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={1773--1780},
  year={2019},
  organization={IEEE}
}
```
