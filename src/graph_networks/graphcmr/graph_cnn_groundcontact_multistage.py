"""
code from 
    https://raw.githubusercontent.com/nkolot/GraphCMR/master/models/graph_cnn.py
     https://github.com/chaneyddtt/Coarse-to-fine-3D-Animal/blob/main/model/graph_hg.py
This file contains the Definition of GraphCNN
GraphCNN includes ResNet50 as a submodule
"""
from __future__ import division

import torch
import torch.nn as nn

# from .resnet import resnet50
import torchvision.models as models


import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.graph_networks.graphcmr.utils_mesh import Mesh
from src.graph_networks.graphcmr.graph_layers import GraphResBlock, GraphLinear


class GraphCNNMS(nn.Module):
    
    def __init__(self, mesh, num_downsample=0, num_layers=5, n_resnet_out=256, num_channels=256):
        '''
        Args:
            mesh: mesh data that store the adjacency matrix
            num_channels: number of channels of GCN
            num_downsample: number of downsampling of the input mesh
        '''
        
        super(GraphCNNMS, self).__init__()

        self.A = mesh._A[num_downsample:] # get the correct adjacency matrix because the input might be downsampled
        # self.num_layers = len(self.A) - 1
        self.num_layers = num_layers
        assert self.num_layers <= len(self.A) - 1
        print("Number of downsampling layer: {}".format(self.num_layers))
        self.num_downsample = num_downsample
        self.n_resnet_out = n_resnet_out

        self.lin1 = GraphLinear(3 + n_resnet_out, 2 * num_channels)
        self.res1 = GraphResBlock(2 * num_channels, num_channels, self.A[0])
        encode_layers = []
        decode_layers = []

        for i in range(self.num_layers + 1):    # range(len(self.A)):
            encode_layers.append(GraphResBlock(num_channels, num_channels, self.A[i]))

            decode_layers.append(GraphResBlock((i+1)*num_channels, (i+1)*num_channels,
                                                   self.A[self.num_layers - i]))
            current_channels = (i+1)*num_channels
            # number of channels for the input is different because of the concatenation operation
        self.n_out_gc = 2       # two labels per vertex  
        self.gc  = nn.Sequential(GraphResBlock(current_channels, 64, self.A[0]),
                                   GraphResBlock(64, 32, self.A[0]),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, self.n_out_gc))

        self.encoder = nn.Sequential(*encode_layers)
        self.decoder = nn.Sequential(*decode_layers)
        self.mesh = mesh




    def forward(self, image_enc):
        """Forward pass
        Inputs:
            image_enc: size = (B, self.n_resnet_out) 
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: size = (B, 3)
        """
        # import pdb; pdb.set_trace()

        batch_size = image_enc.shape[0]
        # ref_vertices = (self.mesh.get_ref_vertices(n=self.num_downsample).t())[None, :, :].expand(batch_size, -1, -1)  # (bs, 3, 973)
        ref_vertices = (self.mesh.ref_vertices.t())[None, :, :].expand(batch_size, -1, -1)  # (bs, 3, 973)
        '''image_resnet = self.resnet(image)       # (bs, 512)'''
        image_enc_prep = image_enc.view(batch_size, -1, 1).expand(-1, -1, ref_vertices.shape[-1]) # (bs, 512, 973)

        # prepare network input
        #   -> for each node we feed the location of the vertex in the template mesh and an image encoding
        x = torch.cat([ref_vertices, image_enc_prep], dim=1)
        x = self.lin1(x)
        x = self.res1(x)
        x_ = [x]
        output_list = []
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
                output_list.append(x)
            else:
                x = self.decoder[i](x)
                output_list.append(x)
                x = self.mesh.upsample(x.transpose(1, 2), n1=self.num_layers-i+self.num_downsample,
                                       n2=self.num_layers-i-1+self.num_downsample)
                x = x.transpose(1, 2)
                x = torch.cat([x, x_[self.num_layers-i-1]], dim=1) # skip connection between encoder and decoder

        ground_contact = self.gc(x)

        return ground_contact, output_list       # , ground_flatness
