"""
code from https://raw.githubusercontent.com/nkolot/GraphCMR/master/models/graph_cnn.py
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


class GraphCNN(nn.Module):
    
    def __init__(self, A, ref_vertices, n_resnet_in, n_resnet_out, num_layers=5, num_channels=512):
        super(GraphCNN, self).__init__()
        self.A = A
        self.ref_vertices = ref_vertices
        # self.resnet = resnet50(pretrained=True)
        #   -> within the GraphCMR network they ignore the last fully connected layer
        # replace the first layer
        self.resnet = models.resnet34(pretrained=False)  
        n_in = 3 + 1
        self.resnet.conv1 = nn.Conv2d(n_resnet_in, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # replace the last layer
        self.resnet.fc = nn.Linear(512, n_resnet_out) 

        layers = [GraphLinear(3 + n_resnet_out, 2 * num_channels)]  # [GraphLinear(3 + 2048, 2 * num_channels)]
        layers.append(GraphResBlock(2 * num_channels, num_channels, A))
        for i in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels, A))
        self.n_out_gc = 2       # two labels per vertex  
        self.gc = nn.Sequential(GraphResBlock(num_channels, 64, A),
                                   GraphResBlock(64, 32, A),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, self.n_out_gc))
        self.gcnn = nn.Sequential(*layers)
        self.n_out_flatground = 1
        self.flat_ground = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
                                      nn.ReLU(inplace=True),
                                      GraphLinear(num_channels, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(A.shape[0], self.n_out_flatground))

    def forward(self, image):
        """Forward pass
        Inputs:
            image: size = (B, 3, 256, 256)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: size = (B, 3)
        """
        # import pdb; pdb.set_trace()

        batch_size = image.shape[0]
        ref_vertices = self.ref_vertices[None, :, :].expand(batch_size, -1, -1)     # (bs, 3, 973)
        image_resnet = self.resnet(image)       # (bs, 512)
        image_enc = image_resnet.view(batch_size, -1, 1).expand(-1, -1, ref_vertices.shape[-1]) # (bs, 512, 973)
        x = torch.cat([ref_vertices, image_enc], dim=1)
        x = self.gcnn(x)        # (bs, 512, 973)
        ground_contact = self.gc(x)      # (bs, 2, 973)
        ground_flatness = self.flat_ground(x).view(batch_size, self.n_out_flatground)    # (bs, 1)
        return ground_contact, ground_flatness
