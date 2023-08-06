
"""
code from: https://github.com/chaneyddtt/Coarse-to-fine-3D-Animal/blob/main/model/graph_hg.py
This file contains the Definition of GraphCNN
GraphCNN includes ResNet50 as a submodule
"""
from __future__ import division

import torch
import torch.nn as nn

from model.networks.graph_layers import GraphResBlock, GraphLinear
from smal.mesh import Mesh
from smal.smal_torch import SMAL

# encoder-decoder structured GCN with skip connections
class GraphCNN_hg(nn.Module):

    def __init__(self, mesh,  num_channels=256, local_feat=False, num_downsample=0):
        '''
        Args:
            mesh: mesh data that store the adjacency matrix
            num_channels: number of channels of GCN
            local_feat: whether use local feature for refinement
            num_downsample: number of downsampling of the input mesh
        '''
        super(GraphCNN_hg, self).__init__()
        self.A = mesh._A[num_downsample:] # get the correct adjacency matrix because the input might be downsampled
        self.num_layers = len(self.A) - 1
        print("Number of downsampling layer: {}".format(self.num_layers))
        self.num_downsample = num_downsample
        if local_feat:
            self.lin1 = GraphLinear(3 + 2048 + 3840, 2 * num_channels)
        else:
            self.lin1 = GraphLinear(3 + 2048, 2 * num_channels)
        self.res1 = GraphResBlock(2 * num_channels, num_channels, self.A[0])
        encode_layers = []
        decode_layers = []

        for i in range(len(self.A)):
            encode_layers.append(GraphResBlock(num_channels, num_channels, self.A[i]))

            decode_layers.append(GraphResBlock((i+1)*num_channels, (i+1)*num_channels,
                                                   self.A[len(self.A) - i - 1]))
            current_channels = (i+1)*num_channels
            # number of channels for the input is different because of the concatenation operation
        self.shape = nn.Sequential(GraphResBlock(current_channels, 64, self.A[0]),
                                   GraphResBlock(64, 32, self.A[0]),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))

        self.encoder = nn.Sequential(*encode_layers)
        self.decoder = nn.Sequential(*decode_layers)
        self.mesh = mesh

    def forward(self, verts_c, img_fea_global, img_fea_multiscale=None, points_local=None):
        '''
        Args:
            verts_c: vertices from the coarse estimation
            img_fea_global: global feature for mesh refinement
            img_fea_multiscale: multi-scale feature from the encoder, used for local feature extraction
            points_local: 2D keypoint for local feature extraction
        Returns: refined mesh
        '''
        batch_size = img_fea_global.shape[0]
        ref_vertices = verts_c.transpose(1, 2)
        image_enc = img_fea_global.view(batch_size, 2048, 1).expand(-1, -1, ref_vertices.shape[-1])
        if points_local is not None:
            feat_local = torch.nn.functional.grid_sample(img_fea_multiscale, points_local)
            x = torch.cat([ref_vertices, image_enc, feat_local.squeeze(2)], dim=1)
        else:
            x = torch.cat([ref_vertices, image_enc], dim=1)
        x = self.lin1(x)
        x = self.res1(x)
        x_ = [x]
        for i in range(self.num_layers + 1):
            if i == self.num_layers:
                x = self.encoder[i](x)
            else:
                x = self.encoder[i](x)
                x = self.mesh.downsample(x.transpose(1, 2), n1=self.num_downsample+i, n2=self.num_downsample+i+1)
                x = x.transpose(1, 2)
                if i < self.num_layers-1:
                    x_.append(x)
        for i in range(self.num_layers + 1):
            if i == self.num_layers:
                x = self.decoder[i](x)
            else:
                x = self.decoder[i](x)
                x = self.mesh.upsample(x.transpose(1, 2), n1=self.num_layers-i+self.num_downsample,
                                       n2=self.num_layers-i-1+self.num_downsample)
                x = x.transpose(1, 2)
                x = torch.cat([x, x_[self.num_layers-i-1]], dim=1) # skip connection between encoder and decoder

        shape = self.shape(x)
        return shape