# code from: https://github.com/chaneyddtt/Coarse-to-fine-3D-Animal/blob/main/util/loss_utils.py

import numpy as np
import torch 


# Laplacian loss, calculate the Laplacian coordiante of both coarse and refined vertices and then compare the difference
class LaplacianCTF(torch.nn.Module):
    def __init__(self, adjmat, device):
        '''
        Args:
            adjmat: adjacency matrix of the input graph data
            device: specify device for training
        '''
        super(LaplacianCTF, self).__init__()
        adjmat.data = np.ones_like(adjmat.data)
        adjmat = torch.from_numpy(adjmat.todense()).float()
        dg = torch.sum(adjmat, dim=-1)
        dg_m = torch.diag(dg)
        ls = dg_m - adjmat
        self.ls = ls.unsqueeze(0).to(device)  # Should be normalized by the diagonal elements according to
                                              # the origial definition, this one also works fine.

    def forward(self, verts_pred, verts_gt, smooth=False):
        verts_pred = torch.matmul(self.ls, verts_pred)
        verts_gt = torch.matmul(self.ls, verts_gt)
        loss = torch.norm(verts_pred - verts_gt, dim=-1).mean()
        if smooth:
            loss_smooth = torch.norm(torch.matmul(self.ls, verts_pred), dim=-1).mean()
            return loss, loss_smooth
        return loss, None


# read the adjacency matrix, which will used in the Laplacian regularizer
# data = np.load('./data/mesh_down_sampling_4.npz', encoding='latin1', allow_pickle=True)
# adjmat = data['A'][0]
# laplacianloss = Laplacian(adjmat, device)
# 
# verts_clone = verts.detach().clone()
# loss_arap, loss_smooth = laplacianloss(verts_refine, verts_clone)
# loss_arap = args.w_arap * loss_arap
#