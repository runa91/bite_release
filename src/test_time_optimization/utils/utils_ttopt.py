
import os
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
from lifting_to_3d.utils.geometry_utils import rot6d_to_rotmat, rotmat_to_rot6d  # , batch_rot2aa, geodesic_loss_R


def reset_loss_values(losses):
    # losses is a dict
    for key, val in losses.items(): 
        val['value'] = 0.0
    return losses

def get_optimed_pose_with_glob(optimed_orient_6d, optimed_pose_6d):
    # optimed_orient_6d: (1, 1, 6)
    # optimed_pose_6d:  (1, 34, 6)
    bs = optimed_pose_6d.shape[0]
    assert bs == 1
    optimed_pose_with_glob_6d = torch.cat((optimed_orient_6d, optimed_pose_6d), dim=1)
    optimed_pose_with_glob = rot6d_to_rotmat(optimed_pose_with_glob_6d.reshape((-1, 6))).reshape((bs, -1, 3, 3))
    return optimed_pose_with_glob



