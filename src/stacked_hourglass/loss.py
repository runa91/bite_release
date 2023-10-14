import torch.nn as nn
import torch
from torch.nn.functional import mse_loss
import torch.nn.functional as F
# for NEW: losses when calculated on keypoint locations
# see https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/subpix/dsnt.html
# from kornia.geometry import dsnt            # old kornia version  
from kornia.geometry.subpix import dsnt     # kornia 0.4.0

def joints_mse_loss_orig(output, target, target_weight=None):
    batch_size = output.size(0)
    num_joints = output.size(1)
    heatmaps_pred = output.view((batch_size, num_joints, -1)).split(1, 1)
    heatmaps_gt = target.view((batch_size, num_joints, -1)).split(1, 1)

    loss = 0
    for idx in range(num_joints):
        heatmap_pred = heatmaps_pred[idx]
        heatmap_gt = heatmaps_gt[idx]
        if target_weight is None:
            loss += 0.5 * mse_loss(heatmap_pred, heatmap_gt, reduction='mean')
        else:
            loss += 0.5 * mse_loss(
                heatmap_pred.mul(target_weight[:, idx]),
                heatmap_gt.mul(target_weight[:, idx]),
                reduction='mean'
            )

    return loss / num_joints


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super().__init__()
        self.use_target_weight = use_target_weight
        raise NotImplementedError

    def forward(self, output, target, target_weight):
        if not self.use_target_weight:
            target_weight = None
        return joints_mse_loss_orig(output, target, target_weight)


def joints_mse_loss_onKPloc(output, target, meta, target_weight=None):
    # debugging:
    # for old kornia version
    # output_softmax_2d = dsnt.spatial_softmax_2d(target, temperature=torch.tensor(100))
    # output_kp = dsnt.spatial_softargmax_2d(output_softmax_2d, normalized_coordinates=False) + 1 
    # print(output_kp[0])
    # print(meta['tpts'][0])
    # render gaussian
    # dsnt.render_gaussian_2d(meta['tpts'][0][0, :2].to('cpu'), torch.tensor(([5., 5.])).to('cpu'), [256, 256], False)
    # output_softmax_2d = dsnt.spatial_softmax_2d(output, temperature=torch.tensor(100))
    # target_norm = target / target.sum(axis=3).sum(axis=2)[:, :, None, None]
    # output_softmax_2d = dsnt.spatial_softmax_2d(output*10)       # (target, temperature=torch.tensor(10))
    # output_kp = dsnt.spatial_softargmax_2d(target_norm, normalized_coordinates=False) + 1 

    # normalize target heatmap
    target_norm = target        # now we have normalized heatmaps

    # normalize predictions -> from logits to probability distribution
    output_norm = dsnt.spatial_softmax2d(output, temperature=torch.tensor(1))

    # heatmap loss (for normalization)
    heatmap_loss = joints_mse_loss_orig(output_norm, target_norm, target_weight)

    # keypoint distance loss (average distance in pixels)
    output_kp = dsnt.spatial_expectation2d(output_norm, normalized_coordinates=False) + 1   # (bs, 20, 2)
    target_kp = meta['tpts'].to(output_kp.device)        # (bs, 20, 3)
    output_kp_resh = output_kp.reshape((-1, 2))
    target_kp_resh = target_kp[:, :, :2].reshape((-1, 2))
    weights_resh = target_kp[:, :, 2].reshape((-1))
    # dist_loss = (((output_kp_resh - target_kp_resh)**2).sum(axis=1).sqrt()*weights_resh)[weights_resh>0].sum() / min(weights_resh[weights_resh>0].sum(), 1e-5)
    dist_loss = (((output_kp_resh - target_kp_resh)[weights_resh>0]**2).sum(axis=1).sqrt()*weights_resh[weights_resh>0]).sum() / max(weights_resh[weights_resh>0].sum(), 1e-5)

    # distlossonly: return dist_loss * 1e-4 
    # both: return dist_loss * 1e-4 + heatmap_loss*100 
    return dist_loss * 1e-4 + heatmap_loss*100 


class JointsMSELoss_onKPloc(nn.Module):
    def __init__(self, use_target_weight=True):
        super().__init__()
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        if not self.use_target_weight:
            target_weight = None
        return joints_mse_loss_onKPloc(output, target, meta, target_weight)


def segmentation_loss(output, meta):
    # output: (6, 2, 64, 64)
    # meta.keys(): ['index', 'center', 'scale', 'pts', 'tpts', 'target_weight', 'breed_index', 'silh']
    # prepare target silhouettes
    target_silh = meta['silh']
    target_silh_l = target_silh.to(torch.long)
    criterion_ce = nn.CrossEntropyLoss()
    if output.shape[2] == 64:
        target_silh_64 = F.adaptive_avg_pool2d(target_silh, (64,64))
        target_silh_64[target_silh_64>0.5] = 1
        target_silh_64[target_silh_64<=0.5] = 0
        target_silh_64_l = target_silh_64.to(torch.long)
        loss_silh_64 = criterion_ce(output, target_silh_64_l)       # 0.7
        return loss_silh_64
    else:
        loss_silh_l = criterion_ce(output, target_silh_l)       # 0.7
        return loss_silh_l        






