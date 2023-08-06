
# code from: https://github.com/chaneyddtt/Coarse-to-fine-3D-Animal/blob/main/util/loss_sdf.py

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import matplotlib.pyplot as plt


def dice_loss(score, target):
    # implemented from paper https://arxiv.org/pdf/1606.04797.pdf
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


class tversky_loss(torch.nn.Module):
    # implemented from https://arxiv.org/pdf/1706.05721.pdf
    def __init__(self, alpha, beta):
        '''
        Args:
            alpha: coefficient for false positive prediction
            beta: coefficient for false negtive prediction
        '''
        super(tversky_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def __call__(self, score, target):
        target = target.float()
        smooth = 1e-5
        tp = torch.sum(score * target)
        fn = torch.sum(target * (1 - score))
        fp = torch.sum((1-target) * score)
        loss = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        loss = 1 - loss
        return loss


def compute_sdf1_1(img_gt, out_shape):
    """
    compute the normalized signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1, 1]
    """

    img_gt = img_gt.astype(np.uint8)

    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
            # ignore background
        for c in range(1, out_shape[1]):
            posmask = img_gt[b]
            negmask = 1-posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b][c] = sdf
            assert np.min(sdf) == -1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))
            assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """

    img_gt = img_gt.astype(np.uint8)

    gt_sdf = np.zeros(out_shape)
    debug = False
    for b in range(out_shape[0]): # batch size
        for c in range(0, out_shape[1]):
            posmask = img_gt[b]
            negmask = 1-posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = negdis - posdis
            sdf[boundary==1] = 0
            gt_sdf[b][c] = sdf
            if debug:
                plt.figure()
                plt.subplot(1, 2, 1), plt.imshow(img_gt[b, 0, :, :]), plt.colorbar()
                plt.subplot(1, 2, 2), plt.imshow(gt_sdf[b, 0, :, :]), plt.colorbar()
                plt.show()

    return gt_sdf


def boundary_loss(output, gt):
    """
    compute boundary loss for binary segmentation
    input: outputs_soft: softmax results,  shape=(b,2,x,y,z)
           gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    adopted from http://proceedings.mlr.press/v102/kervadec19a/kervadec19a.pdf
    """
    multipled = torch.einsum('bcxy, bcxy->bcxy', output, gt)
    bd_loss = multipled.mean()

    return bd_loss