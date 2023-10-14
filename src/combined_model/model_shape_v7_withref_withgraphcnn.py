
import pickle as pkl
import numpy as np
import torchvision.models as models
from torchvision import transforms
import torch
from torch import nn
from torch.nn.parameter import Parameter
from kornia.geometry.subpix import dsnt     # kornia 0.4.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from stacked_hourglass.utils.evaluation import get_preds_soft
from stacked_hourglass import hg1, hg2, hg8
from lifting_to_3d.linear_model import LinearModelComplete, LinearModel      
from lifting_to_3d.inn_model_for_shape import INNForShape
from lifting_to_3d.utils.geometry_utils import rot6d_to_rotmat, rotmat_to_rot6d
from smal_pytorch.smal_model.smal_torch_new import SMAL
from smal_pytorch.renderer.differentiable_renderer import SilhRenderer
from bps_2d.bps_for_segmentation import SegBPS
# from configs.SMAL_configs import SMAL_MODEL_DATA_PATH as SHAPE_PRIOR
from configs.SMAL_configs import SMAL_MODEL_CONFIG
from configs.SMAL_configs import MEAN_DOG_BONE_LENGTHS_NO_RED, VERTEX_IDS_TAIL

# NEW: for graph cnn part
from smal_pytorch.smal_model.smal_torch_new import SMAL
from configs.SMAL_configs import SMAL_MODEL_CONFIG
from graph_networks.graphcmr.utils_mesh import Mesh
from graph_networks.graphcmr.graph_cnn_groundcontact_multistage import GraphCNNMS




class SmallLinear(nn.Module):
    def __init__(self, input_size=64, output_size=30, linear_size=128):
        super(SmallLinear, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(input_size, linear_size)
        self.w2 = nn.Linear(linear_size, linear_size)
        self.w3 = nn.Linear(linear_size, output_size)
    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.relu(y)
        y = self.w2(y)
        y = self.relu(y)
        y = self.w3(y)
        return y


class MyConv1d(nn.Module):
    def __init__(self, input_size=37, output_size=30, start=True):
        super(MyConv1d, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.start = start
        self.weight = Parameter(torch.ones((self.output_size)))
        self.bias = Parameter(torch.zeros((self.output_size)))
    def forward(self, x):
        # pre-processing
        if self.start:
            y = x[:, :self.output_size]
        else:
            y = x[:, -self.output_size:]
        y = y * self.weight[None, :] + self.bias[None, :]
        return y


class ModelShapeAndBreed(nn.Module):
    def __init__(self, smal_model_type, n_betas=10, n_betas_limbs=13, n_breeds=121, n_z=512, structure_z_to_betas='default'):
        super(ModelShapeAndBreed, self).__init__()
        self.n_betas = n_betas
        self.n_betas_limbs = n_betas_limbs   # n_betas_logscale
        self.n_breeds = n_breeds
        self.structure_z_to_betas = structure_z_to_betas
        if self.structure_z_to_betas == '1dconv':
            if not (n_z == self.n_betas+self.n_betas_limbs):
                raise ValueError
        self.smal_model_type = smal_model_type
        # shape branch
        self.resnet = models.resnet34(pretrained=False)  
        # replace the first layer
        n_in = 3 + 1
        self.resnet.conv1 = nn.Conv2d(n_in, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # replace the last layer
        self.resnet.fc = nn.Linear(512, n_z) 
        # softmax
        self.soft_max = torch.nn.Softmax(dim=1)
        # fc network (and other versions) to connect z with betas
        p_dropout = 0.2
        if self.structure_z_to_betas == 'default':
            self.linear_betas = LinearModel(linear_size=1024,     
                                                num_stage=1,
                                                p_dropout=p_dropout, 
                                                input_size=n_z,
                                                output_size=self.n_betas)
            self.linear_betas_limbs = LinearModel(linear_size=1024,    
                                                num_stage=1,
                                                p_dropout=p_dropout, 
                                                input_size=n_z,
                                                output_size=self.n_betas_limbs)
        elif self.structure_z_to_betas == 'lin':
            self.linear_betas = nn.Linear(n_z, self.n_betas)
            self.linear_betas_limbs = nn.Linear(n_z, self.n_betas_limbs)
        elif self.structure_z_to_betas == 'fc_0':
            self.linear_betas = SmallLinear(linear_size=128,     # 1024,
                                                input_size=n_z,
                                                output_size=self.n_betas)
            self.linear_betas_limbs = SmallLinear(linear_size=128,     # 1024,
                                                input_size=n_z,
                                                output_size=self.n_betas_limbs)
        elif structure_z_to_betas == 'fc_1':
            self.linear_betas = LinearModel(linear_size=64,     # 1024,
                                                num_stage=1,
                                                p_dropout=0, 
                                                input_size=n_z,
                                                output_size=self.n_betas)
            self.linear_betas_limbs = LinearModel(linear_size=64,     # 1024,
                                                num_stage=1,
                                                p_dropout=0, 
                                                input_size=n_z,
                                                output_size=self.n_betas_limbs)
        elif self.structure_z_to_betas == '1dconv':
            self.linear_betas = MyConv1d(n_z, self.n_betas, start=True)
            self.linear_betas_limbs = MyConv1d(n_z, self.n_betas_limbs, start=False)
        elif self.structure_z_to_betas == 'inn':
            self.linear_betas_and_betas_limbs = INNForShape(self.n_betas, self.n_betas_limbs, betas_scale=1.0, betas_limbs_scale=1.0)
        else:
            raise ValueError
        # network to connect latent shape vector z with dog breed classification
        self.linear_breeds = LinearModel(linear_size=1024,    # 1024,
                                            num_stage=1,
                                            p_dropout=p_dropout, 
                                            input_size=n_z,
                                            output_size=self.n_breeds)
        # shape multiplicator
        self.shape_multiplicator_np = np.ones(self.n_betas)
        with open(SMAL_MODEL_CONFIG[self.smal_model_type]['smal_model_data_path'], 'rb') as file:
            u = pkl._Unpickler(file)
            u.encoding = 'latin1'
            res = u.load()
        # shape predictions are centered around the mean dog of our dog model
        if 'dog_cluster_mean' in res.keys():
            self.betas_mean_np = res['dog_cluster_mean'] 
        else:
            assert res['cluster_means'].shape[0]==1
            self.betas_mean_np = res['cluster_means'][0, :]

                                        
    def forward(self, img, seg_raw=None, seg_prep=None):
        # img is the network input image 
        # seg_raw is before softmax and subtracting 0.5
        # seg_prep would be the prepared_segmentation
        if seg_prep is None:
            seg_prep = self.soft_max(seg_raw)[:, 1:2, :, :] - 0.5       
        input_img_and_seg = torch.cat((img, seg_prep), axis=1)
        res_output = self.resnet(input_img_and_seg)
        dog_breed_output = self.linear_breeds(res_output) 
        if self.structure_z_to_betas == 'inn':
            shape_output_orig, shape_limbs_output_orig = self.linear_betas_and_betas_limbs(res_output)
        else:
            shape_output_orig = self.linear_betas(res_output) * 0.1
            betas_mean = torch.tensor(self.betas_mean_np).float().to(img.device)
            shape_output = shape_output_orig + betas_mean[None, 0:self.n_betas]
            shape_limbs_output_orig = self.linear_betas_limbs(res_output)
            shape_limbs_output = shape_limbs_output_orig * 0.1
        output_dict = {'z': res_output,
                        'breeds': dog_breed_output,
                        'betas': shape_output_orig,
                        'betas_limbs': shape_limbs_output_orig}
        return output_dict



class LearnableShapedirs(nn.Module):
    def __init__(self, sym_ids_dict, shapedirs_init, n_betas, n_betas_fixed=10):
        super(LearnableShapedirs, self).__init__()
        # shapedirs_init = self.smal.shapedirs.detach()
        self.n_betas = n_betas
        self.n_betas_fixed = n_betas_fixed
        self.sym_ids_dict = sym_ids_dict
        sym_left_ids = self.sym_ids_dict['left']
        sym_right_ids = self.sym_ids_dict['right']
        sym_center_ids = self.sym_ids_dict['center']
        self.n_center = sym_center_ids.shape[0]
        self.n_left = sym_left_ids.shape[0]
        self.n_sd = self.n_betas - self.n_betas_fixed     # number of learnable shapedirs
        # get indices to go from half_shapedirs to shapedirs
        inds_back = np.zeros((3889))
        for ind in range(0, sym_center_ids.shape[0]):
            ind_in_forward = sym_center_ids[ind]
            inds_back[ind_in_forward] = ind
        for ind in range(0, sym_left_ids.shape[0]):
            ind_in_forward = sym_left_ids[ind]
            inds_back[ind_in_forward] = sym_center_ids.shape[0] + ind
        for ind in range(0, sym_right_ids.shape[0]):
            ind_in_forward = sym_right_ids[ind]
            inds_back[ind_in_forward] = sym_center_ids.shape[0] + sym_left_ids.shape[0] + ind
        self.register_buffer('inds_back_torch', torch.Tensor(inds_back).long())
        # self.smal.shapedirs: (51, 11667)
        # shapedirs: (3889, 3, n_sd)
        # shapedirs_half: (2012, 3, n_sd)
        sd = shapedirs_init[:self.n_betas, :].permute((1, 0)).reshape((-1, 3, self.n_betas))
        self.register_buffer('sd', sd)
        sd_center = sd[sym_center_ids, :, self.n_betas_fixed:]
        sd_left = sd[sym_left_ids, :, self.n_betas_fixed:]
        self.register_parameter('learnable_half_shapedirs_c0', torch.nn.Parameter(sd_center[:, 0, :].detach()))
        self.register_parameter('learnable_half_shapedirs_c2', torch.nn.Parameter(sd_center[:, 2, :].detach()))
        self.register_parameter('learnable_half_shapedirs_l0', torch.nn.Parameter(sd_left[:, 0, :].detach()))
        self.register_parameter('learnable_half_shapedirs_l1', torch.nn.Parameter(sd_left[:, 1, :].detach()))
        self.register_parameter('learnable_half_shapedirs_l2', torch.nn.Parameter(sd_left[:, 2, :].detach()))
    def forward(self):
        device = self.learnable_half_shapedirs_c0.device
        half_shapedirs_center = torch.stack((self.learnable_half_shapedirs_c0, \
                                            torch.zeros((self.n_center, self.n_sd)).to(device), \
                                            self.learnable_half_shapedirs_c2), axis=1)
        half_shapedirs_left = torch.stack((self.learnable_half_shapedirs_l0, \
                                            self.learnable_half_shapedirs_l1, \
                                            self.learnable_half_shapedirs_l2), axis=1)
        half_shapedirs_right = torch.stack((self.learnable_half_shapedirs_l0, \
                                            - self.learnable_half_shapedirs_l1, \
                                            self.learnable_half_shapedirs_l2), axis=1)
        half_shapedirs_tot = torch.cat((half_shapedirs_center, half_shapedirs_left, half_shapedirs_right))
        shapedirs = torch.index_select(half_shapedirs_tot, dim=0, index=self.inds_back_torch)
        shapedirs_complete = torch.cat((self.sd[:, :, :self.n_betas_fixed], shapedirs), axis=2)      # (3889, 3, n_sd)
        shapedirs_complete_prepared = torch.cat((self.sd[:, :, :10], shapedirs), axis=2).reshape((-1, 30)).permute((1, 0))   # (n_sd, 11667)
        return shapedirs_complete, shapedirs_complete_prepared


class ModelRefinement(nn.Module):
    def __init__(self, n_betas=10, n_betas_limbs=7, n_breeds=121, n_keyp=20, n_joints=35, ref_net_type='add', graphcnn_type='inexistent', isflat_type='inexistent', shaperef_type='inexistent'):
        super(ModelRefinement, self).__init__()
        self.n_betas = n_betas
        self.n_betas_limbs = n_betas_limbs
        self.n_breeds = n_breeds
        self.n_keyp = n_keyp
        self.n_joints = n_joints
        self.n_out_seg = 256
        self.n_out_keyp = 256
        self.n_out_enc = 256
        self.linear_size = 1024
        self.linear_size_small = 128
        self.ref_net_type = ref_net_type
        self.graphcnn_type = graphcnn_type
        self.isflat_type = isflat_type
        self.shaperef_type = shaperef_type
        p_dropout = 0.2
        # --- segmentation encoder
        if self.ref_net_type in ['multrot_res34', 'multrot01all_res34']:
            self.ref_res = models.resnet34(pretrained=False)
        else:
            self.ref_res = models.resnet18(pretrained=False)
        # replace the first layer
        self.ref_res.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # replace the last layer
        self.ref_res.fc = nn.Linear(512, self.n_out_seg) 
        # softmax
        self.soft_max = torch.nn.Softmax(dim=1)
        # --- keypoint encoder
        self.linear_keyp = LinearModel(linear_size=self.linear_size,
                                            num_stage=1,
                                            p_dropout=p_dropout, 
                                            input_size=n_keyp*2*2,
                                            output_size=self.n_out_keyp)
        # --- decoder
        self.linear_combined = LinearModel(linear_size=self.linear_size,
                                            num_stage=1,
                                            p_dropout=p_dropout, 
                                            input_size=self.n_out_seg+self.n_out_keyp,
                                            output_size=self.n_out_enc)
        # output info
        pose = {'name': 'pose', 'n': self.n_joints*6, 'out_shape':[self.n_joints, 6]}
        trans = {'name': 'trans_notnorm', 'n': 3}
        cam = {'name': 'flength_notnorm', 'n': 1}
        betas = {'name': 'betas', 'n': self.n_betas}
        betas_limbs = {'name': 'betas_limbs', 'n': self.n_betas_limbs}
        if self.shaperef_type=='inexistent':          
            self.output_info = [pose, trans, cam]   # , betas]
        else:
            self.output_info = [pose, trans, cam, betas, betas_limbs]
        # output branches
        self.output_info_linear_models = []
        for ind_el, element in enumerate(self.output_info):
            n_in = self.n_out_enc + element['n']
            self.output_info_linear_models.append(LinearModel(linear_size=self.linear_size,
                                    num_stage=1,
                                    p_dropout=p_dropout, 
                                    input_size=n_in,
                                    output_size=element['n']))
            element['linear_model_index'] = ind_el
        self.output_info_linear_models = nn.ModuleList(self.output_info_linear_models)
        # new: predict if the ground is flat
        if not self.isflat_type=='inexistent':          
            self.linear_isflat = LinearModel(linear_size=self.linear_size_small,
                                    num_stage=1,
                                    p_dropout=p_dropout, 
                                    input_size=self.n_out_enc,
                                    output_size=2) # answer is just yes or no

    
        # new for ground contact prediction: graph cnn
        if not self.graphcnn_type=='inexistent':
            num_downsampling = 1
            smal_model_type = '39dogs_norm'
            smal = SMAL(smal_model_type=smal_model_type, template_name='neutral')     
            ROOT_smal_downsampling = os.path.join(os.path.dirname(__file__), './../../data/graphcmr_data/')
            smal_downsampling_npz_name = 'mesh_downsampling_' + os.path.basename(SMAL_MODEL_CONFIG[smal_model_type]['smal_model_path']).replace('.pkl', '_template.npz')
            smal_downsampling_npz_path = ROOT_smal_downsampling + smal_downsampling_npz_name  # 'data/mesh_downsampling.npz'
            self.my_custom_smal_dog_mesh = Mesh(filename=smal_downsampling_npz_path, num_downsampling=num_downsampling, nsize=1, body_model=smal) # , device=device)
            # create GraphCNN
            num_layers = 2  # <= len(my_custom_mesh._A)-1
            n_resnet_out = self.n_out_enc       # 256
            num_channels = 256      # 512 
            self.graph_cnn = GraphCNNMS(mesh=self.my_custom_smal_dog_mesh, 
                                num_downsample = num_downsampling, 
                                num_layers = num_layers, 
                                n_resnet_out = n_resnet_out, 
                                num_channels = num_channels)    # .to(device)
        
    
    
    def forward(self, keyp_sh, keyp_pred, in_pose_3x3, in_trans_notnorm, in_cam_notnorm, in_betas, in_betas_limbs, seg_pred_prep=None, seg_sh_raw=None, seg_sh_prep=None):
        # img is the network input image 
        # seg_raw is before softmax and subtracting 0.5
        # seg_prep would be the prepared_segmentation
        batch_size = in_pose_3x3.shape[0]
        device = in_pose_3x3.device
        dtype = in_pose_3x3.dtype
        # --- segmentation encoder
        if seg_sh_prep is None:
            seg_sh_prep = self.soft_max(seg_sh_raw)[:, 1:2, :, :] - 0.5       # class 1 is the dog
        input_seg_conc = torch.cat((seg_sh_prep, seg_pred_prep), axis=1)  
        network_output_seg = self.ref_res(input_seg_conc)
        # --- keypoint encoder
        keyp_conc = torch.cat((keyp_sh.reshape((-1, keyp_sh.shape[1]*keyp_sh.shape[2])), keyp_pred.reshape((-1, keyp_sh.shape[1]*keyp_sh.shape[2]))), axis=1) 
        network_output_keyp = self.linear_keyp(keyp_conc)
        # --- decoder
        x = torch.cat((network_output_seg, network_output_keyp), axis=1)
        y_comb = self.linear_combined(x)
        in_pose_6d = rotmat_to_rot6d(in_pose_3x3.reshape((-1, 3, 3))).reshape((in_pose_3x3.shape[0], -1, 6))
        in_dict = {'pose': in_pose_6d,
                    'trans_notnorm': in_trans_notnorm,
                    'flength_notnorm': in_cam_notnorm, 
                    'betas': in_betas, 
                    'betas_limbs': in_betas_limbs}
        results = {}
        for element in self.output_info:

            linear_model = self.output_info_linear_models[element['linear_model_index']]
            y = torch.cat((y_comb, in_dict[element['name']].reshape((-1, element['n']))), axis=1)
            if 'out_shape' in element.keys():
                if element['name'] == 'pose':
                    if self.ref_net_type in ['multrot', 'multrot01', 'multrot01all', 'multrotxx', 'multrot_res34', 'multrot01all_res34']:      # if self.ref_net_type == 'multrot' or self.ref_net_type == 'multrot_res34':
                        #   multiply the rotations with each other -> just predict a correction
                        #   the correction should be initialized as identity
                        # res_pose_out = (linear_model(y)).reshape((-1, element['out_shape'][0], element['out_shape'][1])) + in_dict[element['name']]
                        identity_rot6d = torch.tensor(([1., 0., 0., 1., 0., 0.])).repeat((in_pose_3x3.shape[0]*in_pose_3x3.shape[1], 1)).to(device=device, dtype=dtype)
                        if self.ref_net_type in ['multrot01', 'multrot01all', 'multrot01all_res34']:
                            res_pose_out = identity_rot6d + 0.1*(linear_model(y)).reshape((-1, element['out_shape'][1]))  
                        elif self.ref_net_type == 'multrotxx':
                            res_pose_out = identity_rot6d + 0.0*(linear_model(y)).reshape((-1, element['out_shape'][1]))    
                        else:
                            res_pose_out = identity_rot6d + (linear_model(y)).reshape((-1, element['out_shape'][1]))    
                        res_pose_rotmat = rot6d_to_rotmat(res_pose_out.reshape((-1, 6)))    # (bs*35, 3, 3)     .reshape((batch_size, -1, 3, 3))
                        res_tot_rotmat = torch.bmm(res_pose_rotmat.reshape((-1, 3, 3)), in_pose_3x3.reshape((-1, 3, 3))).reshape((batch_size, -1, 3, 3))   # (bs, 5, 3, 3)
                        results['pose_rotmat'] = res_tot_rotmat
                    elif self.ref_net_type == 'add':
                        res_6d = (linear_model(y)).reshape((-1, element['out_shape'][0], element['out_shape'][1])) + in_dict['pose']
                        results['pose_rotmat'] = rot6d_to_rotmat(res_6d.reshape((-1, 6))).reshape((batch_size, -1, 3, 3)) 
                    else:
                        raise ValueError
                else:
                    if self.ref_net_type in ['multrot01all', 'multrot01all_res34']:
                        results[element['name']] = (0.1*linear_model(y)).reshape((-1, element['out_shape'][0], element['out_shape'][1])) + in_dict[element['name']]
                    else:
                        results[element['name']] = (linear_model(y)).reshape((-1, element['out_shape'][0], element['out_shape'][1])) + in_dict[element['name']]
            else:
                if self.ref_net_type in ['multrot01all', 'multrot01all_res34']:
                    results[element['name']] = 0.1*linear_model(y) + in_dict[element['name']]     
                else:
                    results[element['name']] = linear_model(y) + in_dict[element['name']]  

        # add prediction if ground is flat
        if not self.isflat_type=='inexistent':
            isflat = self.linear_isflat(y_comb)
            results['isflat'] = isflat

        # add graph cnn
        if not self.graphcnn_type=='inexistent':
            ground_contact_downsampled, ground_cntact_all_stages_output = self.graph_cnn(y_comb)
            ground_contact = self.my_custom_smal_dog_mesh.upsample(ground_contact_downsampled.transpose(1,2))
            results['vertexwise_ground_contact'] = ground_contact

        return results




class ModelImageToBreed(nn.Module):
    def __init__(self, smal_model_type, arch='hg8', n_joints=35, n_classes=20, n_partseg=15, n_keyp=20, n_bones=24, n_betas=10, n_betas_limbs=7, n_breeds=121, image_size=256, n_z=512, thr_keyp_sc=None, add_partseg=True):
        super(ModelImageToBreed, self).__init__()
        self.n_classes = n_classes
        self.n_partseg = n_partseg
        self.n_betas = n_betas
        self.n_betas_limbs = n_betas_limbs
        self.n_keyp = n_keyp
        self.n_bones = n_bones
        self.n_breeds = n_breeds
        self.image_size = image_size
        self.upsample_seg = True
        self.threshold_scores = thr_keyp_sc 
        self.n_z = n_z
        self.add_partseg = add_partseg
        self.smal_model_type = smal_model_type
        # ------------------------------ STACKED HOUR GLASS ------------------------------        
        if arch == 'hg8':
            self.stacked_hourglass = hg8(pretrained=False, num_classes=self.n_classes, num_partseg=self.n_partseg, upsample_seg=self.upsample_seg, add_partseg=self.add_partseg)
        else:
            raise Exception('unrecognised model architecture: ' + arch)
        # ------------------------------ SHAPE AND BREED MODEL ------------------------------
        self.breed_model = ModelShapeAndBreed(smal_model_type=self.smal_model_type, n_betas=self.n_betas, n_betas_limbs=self.n_betas_limbs, n_breeds=self.n_breeds, n_z=self.n_z)
    def forward(self, input_img, norm_dict=None, bone_lengths_prepared=None, betas=None):
        batch_size = input_img.shape[0]
        device = input_img.device
        # ------------------------------ STACKED HOUR GLASS ------------------------------
        hourglass_out_dict = self.stacked_hourglass(input_img)
        last_seg = hourglass_out_dict['seg_final']
        last_heatmap = hourglass_out_dict['out_list_kp'][-1] 
        # - prepare keypoints (from heatmap)
        # normalize predictions -> from logits to probability distribution
        # last_heatmap_norm = dsnt.spatial_softmax2d(last_heatmap, temperature=torch.tensor(1))
        # keypoints = dsnt.spatial_expectation2d(last_heatmap_norm, normalized_coordinates=False) + 1   # (bs, 20, 2)
        # keypoints_norm = dsnt.spatial_expectation2d(last_heatmap_norm, normalized_coordinates=True)    # (bs, 20, 2)
        keypoints_norm, scores = get_preds_soft(last_heatmap, return_maxval=True, norm_coords=True)
        if self.threshold_scores is not None:
            scores[scores>self.threshold_scores] = 1.0
            scores[scores<=self.threshold_scores] = 0.0
        # ------------------------------ SHAPE AND BREED MODEL ------------------------------
        # breed_model takes as input the image as well as the predicted segmentation map 
        #     -> we need to split up ModelImageTo3d, such that we can use the silhouette
        resnet_output = self.breed_model(img=input_img, seg_raw=last_seg)
        pred_breed = resnet_output['breeds']       # (bs, n_breeds)
        pred_betas = resnet_output['betas']
        pred_betas_limbs = resnet_output['betas_limbs']
        small_output = {'keypoints_norm': keypoints_norm,
                        'keypoints_scores': scores}
        small_output_reproj = {'betas': pred_betas,
                                'betas_limbs': pred_betas_limbs,
                                'dog_breed': pred_breed}
        return small_output, None, small_output_reproj

class ModelImageTo3d_withshape_withproj(nn.Module):
    def __init__(self, smal_model_type, smal_keyp_conf=None, arch='hg8', num_stage_comb=2, num_stage_heads=1, num_stage_heads_pose=1, trans_sep=False, n_joints=35, n_classes=20, n_partseg=15, n_keyp=20, n_bones=24, n_betas=10, n_betas_limbs=6, n_breeds=121, image_size=256, n_z=512, n_segbps=64*2, thr_keyp_sc=None, add_z_to_3d_input=True, add_segbps_to_3d_input=False, add_partseg=True, silh_no_tail=True, fix_flength=False, render_partseg=False, structure_z_to_betas='default', structure_pose_net='default', nf_version=None, ref_net_type='add', ref_detach_shape=True, graphcnn_type='inexistent', isflat_type='inexistent', shaperef_type='inexistent'):
        super(ModelImageTo3d_withshape_withproj, self).__init__()
        self.n_classes = n_classes
        self.n_partseg = n_partseg
        self.n_betas = n_betas
        self.n_betas_limbs = n_betas_limbs
        self.n_keyp = n_keyp
        self.n_joints = n_joints
        self.n_bones = n_bones
        self.n_breeds = n_breeds
        self.image_size = image_size
        self.threshold_scores = thr_keyp_sc 
        self.upsample_seg = True
        self.silh_no_tail = silh_no_tail
        self.add_z_to_3d_input = add_z_to_3d_input       
        self.add_segbps_to_3d_input = add_segbps_to_3d_input
        self.add_partseg = add_partseg
        self.ref_net_type = ref_net_type
        self.ref_detach_shape = ref_detach_shape
        self.graphcnn_type = graphcnn_type
        self.isflat_type = isflat_type
        self.shaperef_type = shaperef_type
        assert (not self.add_segbps_to_3d_input) or (not self.add_z_to_3d_input)
        self.n_z = n_z   
        if add_segbps_to_3d_input:
            self.n_segbps = n_segbps    # 64
            self.segbps_model = SegBPS()
        else:
            self.n_segbps = 0
        self.fix_flength = fix_flength
        self.render_partseg = render_partseg
        self.structure_z_to_betas = structure_z_to_betas
        self.structure_pose_net = structure_pose_net
        assert self.structure_pose_net in ['default', 'vae', 'normflow']
        self.nf_version = nf_version
        self.smal_model_type = smal_model_type
        assert (smal_keyp_conf is not None)
        self.smal_keyp_conf = smal_keyp_conf
        self.register_buffer('betas_zeros', torch.zeros((1, self.n_betas)))
        self.register_buffer('mean_dog_bone_lengths', torch.tensor(MEAN_DOG_BONE_LENGTHS_NO_RED, dtype=torch.float32))
        p_dropout = 0.2      # 0.5     
        # ------------------------------ SMAL MODEL ------------------------------
        self.smal = SMAL(smal_model_type=self.smal_model_type, template_name='neutral')     
        print('SMAL model type: ' + self.smal.smal_model_type)      
        # New for rendering without tail
        f_np = self.smal.faces.detach().cpu().numpy()
        self.f_no_tail_np = f_np[np.isin(f_np[:,:], VERTEX_IDS_TAIL).sum(axis=1)==0, :]
        # in theory we could optimize for improved shapedirs, but we do not do that
        #   -> would need to implement regularizations 
        #   -> there are better ways than changing the shapedirs
        self.model_learnable_shapedirs = LearnableShapedirs(self.smal.sym_ids_dict, self.smal.shapedirs.detach(), self.n_betas, 10)
        # ------------------------------ STACKED HOUR GLASS ------------------------------        
        if arch == 'hg8':
            self.stacked_hourglass = hg8(pretrained=False, num_classes=self.n_classes, num_partseg=self.n_partseg, upsample_seg=self.upsample_seg, add_partseg=self.add_partseg)
        else:
            raise Exception('unrecognised model architecture: ' + arch)
        # ------------------------------ SHAPE AND BREED MODEL ------------------------------
        self.breed_model = ModelShapeAndBreed(self.smal_model_type, n_betas=self.n_betas, n_betas_limbs=self.n_betas_limbs, n_breeds=self.n_breeds, n_z=self.n_z, structure_z_to_betas=self.structure_z_to_betas)
        # ------------------------------ LINEAR 3D MODEL ------------------------------
        # 3d model -> from image to 3d parameters {2d keypoints from heatmap, pose, trans, flength}
        self.soft_max = torch.nn.Softmax(dim=1)
        input_size = self.n_keyp*3 + self.n_bones
        self.model_3d = LinearModelComplete(linear_size=1024,
                    num_stage_comb=num_stage_comb,
                    num_stage_heads=num_stage_heads,
                    num_stage_heads_pose=num_stage_heads_pose,
                    trans_sep=trans_sep, 
                    p_dropout=p_dropout,        # 0.5, 
                    input_size=input_size,
                    intermediate_size=1024,
                    output_info=None,
                    n_joints=self.n_joints,
                    n_z=self.n_z,
                    add_z_to_3d_input=self.add_z_to_3d_input,
                    n_segbps=self.n_segbps,
                    add_segbps_to_3d_input=self.add_segbps_to_3d_input, 
                    structure_pose_net=self.structure_pose_net,
                    nf_version = self.nf_version)
        # ------------------------------ RENDERING ------------------------------
        self.silh_renderer = SilhRenderer(image_size) 
        # ------------------------------ REFINEMENT -----------------------------
        self.refinement_model = ModelRefinement(n_betas=self.n_betas, n_betas_limbs=self.n_betas_limbs, n_breeds=self.n_breeds, n_keyp=self.n_keyp, n_joints=self.n_joints, ref_net_type=self.ref_net_type, graphcnn_type=self.graphcnn_type, isflat_type=self.isflat_type, shaperef_type=self.shaperef_type)


    def forward(self, input_img, norm_dict=None, bone_lengths_prepared=None, betas=None):
        batch_size = input_img.shape[0]
        device = input_img.device
        # ------------------------------ STACKED HOUR GLASS ------------------------------
        hourglass_out_dict = self.stacked_hourglass(input_img)
        last_seg = hourglass_out_dict['seg_final']
        last_heatmap = hourglass_out_dict['out_list_kp'][-1] 
        # - prepare keypoints (from heatmap)
        # normalize predictions -> from logits to probability distribution
        # last_heatmap_norm = dsnt.spatial_softmax2d(last_heatmap, temperature=torch.tensor(1))
        # keypoints = dsnt.spatial_expectation2d(last_heatmap_norm, normalized_coordinates=False) + 1   # (bs, 20, 2)
        # keypoints_norm = dsnt.spatial_expectation2d(last_heatmap_norm, normalized_coordinates=True)    # (bs, 20, 2)
        keypoints_norm, scores = get_preds_soft(last_heatmap, return_maxval=True, norm_coords=True)
        if self.threshold_scores is not None:
            scores[scores>self.threshold_scores] = 1.0
            scores[scores<=self.threshold_scores] = 0.0
        # ------------------------------ LEARNABLE SHAPE MODEL ------------------------------
        # in our cvpr 2022 paper we do not change the shapedirs
        # learnable_sd_complete has shape (3889, 3, n_sd)
        # learnable_sd_complete_prepared has shape (n_sd, 11667)
        learnable_sd_complete, learnable_sd_complete_prepared = self.model_learnable_shapedirs()
        shapedirs_sel = learnable_sd_complete_prepared        # None
        # ------------------------------ SHAPE AND BREED MODEL ------------------------------
        # breed_model takes as input the image as well as the predicted segmentation map 
        #     -> we need to split up ModelImageTo3d, such that we can use the silhouette
        resnet_output = self.breed_model(img=input_img, seg_raw=last_seg)
        pred_breed = resnet_output['breeds']       # (bs, n_breeds)
        pred_z = resnet_output['z']
        # - prepare shape
        pred_betas = resnet_output['betas']     
        pred_betas_limbs = resnet_output['betas_limbs'] 
        # - calculate bone lengths
        with torch.no_grad():
            use_mean_bone_lengths = False
            if use_mean_bone_lengths:
                bone_lengths_prepared = torch.cat(batch_size*[self.mean_dog_bone_lengths.reshape((1, -1))])
            else:
                assert (bone_lengths_prepared is None)
                bone_lengths_prepared = self.smal.caclulate_bone_lengths(pred_betas, pred_betas_limbs, shapedirs_sel=shapedirs_sel, short=True)
        # ------------------------------ LINEAR 3D MODEL ------------------------------
        # 3d model -> from image to 3d parameters {2d keypoints from heatmap, pose, trans, flength}
        # prepare input for 2d-to-3d network
        keypoints_prepared = torch.cat((keypoints_norm, scores), axis=2)
        if bone_lengths_prepared is None:
            bone_lengths_prepared = torch.cat(batch_size*[self.mean_dog_bone_lengths.reshape((1, -1))])
        # should we add silhouette to 3d input? should we add z?
        if self.add_segbps_to_3d_input:
            seg_raw = last_seg
            seg_prep_bps = self.soft_max(seg_raw)[:, 1, :, :] # class 1 is the dog
            with torch.no_grad():
                seg_prep_np = seg_prep_bps.detach().cpu().numpy()
                bps_output_np = self.segbps_model.calculate_bps_points_batch(seg_prep_np)  # (bs, 64, 2)
                bps_output = torch.tensor(bps_output_np, dtype=torch.float32).to(device).reshape((batch_size, -1))
                bps_output_prep = bps_output * 2. - 1
            input_vec_keyp_bones = torch.cat((keypoints_prepared.reshape((batch_size, -1)), bone_lengths_prepared), axis=1)  
            input_vec = torch.cat((input_vec_keyp_bones, bps_output_prep), dim=1)
        elif self.add_z_to_3d_input:
            # we do not use this in our cvpr 2022 version
            input_vec_keyp_bones = torch.cat((keypoints_prepared.reshape((batch_size, -1)), bone_lengths_prepared), axis=1)  
            input_vec_additional = pred_z       
            input_vec = torch.cat((input_vec_keyp_bones, input_vec_additional), dim=1)
        else:
            input_vec = torch.cat((keypoints_prepared.reshape((batch_size, -1)), bone_lengths_prepared), axis=1)  
        # predict 3d parameters (those are normalized, we need to correct mean and std in a next step)
        output = self.model_3d(input_vec)      
        # add predicted keypoints to the output dict
        output['keypoints_norm'] = keypoints_norm
        output['keypoints_scores'] = scores
        # add predicted segmentation to output dictc
        output['seg_hg'] = hourglass_out_dict['seg_final']
        # - denormalize 3d parameters -> so far predictions were normalized, now we denormalize them again
        pred_trans = output['trans'] * norm_dict['trans_std'][None, :] + norm_dict['trans_mean'][None, :]    # (bs, 3)
        if  self.structure_pose_net == 'default':
            pred_pose_rot6d = output['pose'] + norm_dict['pose_rot6d_mean'][None, :]
        elif self.structure_pose_net == 'normflow':
            pose_rot6d_mean_zeros = torch.zeros_like(norm_dict['pose_rot6d_mean'][None, :])
            pose_rot6d_mean_zeros[:, 0, :] = norm_dict['pose_rot6d_mean'][None, 0, :]
            pred_pose_rot6d = output['pose'] + pose_rot6d_mean_zeros
        else:
            pose_rot6d_mean_zeros = torch.zeros_like(norm_dict['pose_rot6d_mean'][None, :])
            pose_rot6d_mean_zeros[:, 0, :] = norm_dict['pose_rot6d_mean'][None, 0, :]
            pred_pose_rot6d = output['pose'] + pose_rot6d_mean_zeros
        pred_pose_reshx33 = rot6d_to_rotmat(pred_pose_rot6d.reshape((-1, 6)))
        pred_pose = pred_pose_reshx33.reshape((batch_size, -1, 3, 3))
        pred_pose_rot6d = rotmat_to_rot6d(pred_pose_reshx33).reshape((batch_size, -1, 6))

        if self.fix_flength:
            output['flength'] = torch.zeros_like(output['flength'])
            pred_flength = torch.ones_like(output['flength'])*2100  # norm_dict['flength_mean'][None, :]
        else:
            pred_flength_orig = output['flength'] * norm_dict['flength_std'][None, :] + norm_dict['flength_mean'][None, :]   # (bs, 1)
            pred_flength = pred_flength_orig.clone()  # torch.abs(pred_flength_orig)
            pred_flength[pred_flength_orig<=0] = norm_dict['flength_mean'][None, :]

        # ------------------------------ RENDERING ------------------------------
        # get 3d model (SMAL)
        V, keyp_green_3d, _ = self.smal(beta=pred_betas, betas_limbs=pred_betas_limbs, pose=pred_pose, trans=pred_trans, get_skin=True, keyp_conf=self.smal_keyp_conf, shapedirs_sel=shapedirs_sel)
        keyp_3d = keyp_green_3d[:, :self.n_keyp, :]     # (bs, 20, 3)
        # render silhouette
        faces_prep = self.smal.faces.unsqueeze(0).expand((batch_size, -1, -1))
        if not self.silh_no_tail:
            pred_silh_images, pred_keyp = self.silh_renderer(vertices=V, 
                points=keyp_3d, faces=faces_prep, focal_lengths=pred_flength)
        else:
            faces_no_tail_prep = torch.tensor(self.f_no_tail_np).to(device).expand((batch_size, -1, -1))
            pred_silh_images, pred_keyp = self.silh_renderer(vertices=V, 
                points=keyp_3d, faces=faces_no_tail_prep, focal_lengths=pred_flength)
        # get torch 'Meshes'
        torch_meshes = self.silh_renderer.get_torch_meshes(vertices=V, faces=faces_prep) 

        #  render body parts (not part of cvpr 2022 version)
        if self.render_partseg:
            raise NotImplementedError
        else:
            partseg_images = None
            partseg_images_hg = None


        # ------------------------------ REFINEMENT MODEL ------------------------------

        # refinement model
        pred_keyp_norm = (pred_keyp.detach() / (self.image_size - 1) - 0.5)*2
        '''output_ref = self.refinement_model(keypoints_norm.detach(), pred_keyp_norm, \
                            seg_sh_raw=last_seg[:, :, :, :].detach(), seg_pred_prep=pred_silh_images[:, :, :, :].detach()-0.5, \
                            in_pose=output['pose'].detach(), in_trans=output['trans'].detach(), in_cam=output['flength'].detach(), in_betas=pred_betas.detach())'''
        output_ref = self.refinement_model(keypoints_norm.detach(), pred_keyp_norm, \
                            seg_sh_raw=last_seg[:, :, :, :].detach(), seg_pred_prep=pred_silh_images[:, :, :, :].detach()-0.5, \
                            in_pose_3x3=pred_pose.detach(), in_trans_notnorm=output['trans'].detach(), in_cam_notnorm=output['flength'].detach(), in_betas=pred_betas.detach(), in_betas_limbs=pred_betas_limbs.detach())
        # a better alternative would be to submit pred_pose_reshx33

        # nothing changes for betas or shapedirs or z  (should probably not be detached in the end)
        if self.shaperef_type == 'inexistent':
            if self.ref_detach_shape:
                output_ref['betas'] = pred_betas.detach()
                output_ref['betas_limbs'] = pred_betas_limbs.detach()
                output_ref['z'] = pred_z.detach()
                output_ref['shapedirs'] = shapedirs_sel.detach()
            else:
                output_ref['betas'] = pred_betas
                output_ref['betas_limbs'] = pred_betas_limbs
                output_ref['z'] = pred_z
                output_ref['shapedirs'] = shapedirs_sel
        else:
            assert ('betas' in output_ref.keys())
            assert ('betas_limbs' in output_ref.keys())
            output_ref['shapedirs'] = shapedirs_sel     

        # we denormalize flength and trans, but pose is handled differently
        if self.fix_flength:
            output_ref['flength_notnorm'] = torch.zeros_like(output['flength'])
            ref_pred_flength = torch.ones_like(output['flength_notnorm'])*2100  # norm_dict['flength_mean'][None, :]
            raise ValueError    # not sure if we want to have a fixed flength in refinement
        else:
            ref_pred_flength_orig = output_ref['flength_notnorm'] * norm_dict['flength_std'][None, :] + norm_dict['flength_mean'][None, :]   # (bs, 1)
            ref_pred_flength = ref_pred_flength_orig.clone()  # torch.abs(pred_flength_orig)
            ref_pred_flength[ref_pred_flength_orig<=0] = norm_dict['flength_mean'][None, :]
        ref_pred_trans = output_ref['trans_notnorm'] * norm_dict['trans_std'][None, :] + norm_dict['trans_mean'][None, :]    # (bs, 3)

        ref_pred_pose_reshx33 = output_ref['pose_rotmat'].reshape((batch_size, -1, 3, 3))
        ref_pred_pose_rot6d = rotmat_to_rot6d(ref_pred_pose_reshx33.reshape((-1, 3, 3))).reshape((batch_size, -1, 6))

        ref_V, ref_keyp_green_3d, _ = self.smal(beta=output_ref['betas'], betas_limbs=output_ref['betas_limbs'], 
                                        pose=ref_pred_pose_reshx33, trans=ref_pred_trans, get_skin=True, keyp_conf=self.smal_keyp_conf, 
                                        shapedirs_sel=output_ref['shapedirs'])
        ref_keyp_3d = ref_keyp_green_3d[:, :self.n_keyp, :]     # (bs, 20, 3)

        if not self.silh_no_tail:
            faces_prep = self.smal.faces.unsqueeze(0).expand((batch_size, -1, -1))
            ref_pred_silh_images, ref_pred_keyp = self.silh_renderer(vertices=ref_V, 
                points=ref_keyp_3d, faces=faces_prep, focal_lengths=ref_pred_flength)
        else:
            faces_no_tail_prep = torch.tensor(self.f_no_tail_np).to(device).expand((batch_size, -1, -1))
            ref_pred_silh_images, ref_pred_keyp = self.silh_renderer(vertices=ref_V, 
                points=ref_keyp_3d, faces=faces_no_tail_prep, focal_lengths=ref_pred_flength)

        output_ref_unnorm = {'vertices_smal': ref_V,
                            'keyp_3d': ref_keyp_3d,
                            'keyp_2d': ref_pred_keyp,
                            'silh': ref_pred_silh_images,
                            'trans': ref_pred_trans,
                            'flength': ref_pred_flength,
                            'betas': output_ref['betas'],
                            'betas_limbs': output_ref['betas_limbs'],
                            # 'z': output_ref['z'],
                            'pose_rot6d': ref_pred_pose_rot6d,   
                            'pose_rotmat':  ref_pred_pose_reshx33} 
                            # 'shapedirs': shapedirs_sel}

        if not self.graphcnn_type == 'inexistent':
            output_ref_unnorm['vertexwise_ground_contact'] = output_ref['vertexwise_ground_contact']
        if not self.isflat_type=='inexistent':
            output_ref_unnorm['isflat'] = output_ref['isflat']
        if self.shaperef_type == 'inexistent':
            output_ref_unnorm['z'] = output_ref['z']

        # REMARK: we will want to have the predicted differences, for pose this would 
        #   be a rotation matrix, ...
        #       -> TODO: adjust output_orig_ref_comparison
        output_orig_ref_comparison = {#'pose': output['pose'].detach(),
                                    #'trans': output['trans'].detach(),
                                    #'flength': output['flength'].detach(),
                                    # 'pose': output['pose'],
                                    'old_pose_rotmat': pred_pose_reshx33,
                                    'old_trans_notnorm': output['trans'],
                                    'old_flength_notnorm': output['flength'],
                                    # 'ref_pose': output_ref['pose'],
                                    'ref_pose_rotmat': ref_pred_pose_reshx33,
                                    'ref_trans_notnorm': output_ref['trans_notnorm'],
                                    'ref_flength_notnorm': output_ref['flength_notnorm']}



        # ------------------------------ PREPARE OUTPUT ------------------------------
        # create output dictionarys
        # output: contains all output from model_image_to_3d
        # output_unnorm: same as output, but normalizations are undone
        # output_reproj: smal output and reprojected keypoints as well as silhouette 
        keypoints_heatmap_256 = (output['keypoints_norm'] / 2. + 0.5) * (self.image_size - 1)
        output_unnorm = {'pose_rotmat': pred_pose,
                        'flength': pred_flength,
                        'trans': pred_trans,
                        'keypoints':keypoints_heatmap_256}
        output_reproj = {'vertices_smal': V,
                        'torch_meshes': torch_meshes,
                        'keyp_3d': keyp_3d,
                        'keyp_2d': pred_keyp,
                        'silh': pred_silh_images,
                        'betas': pred_betas,
                        'betas_limbs': pred_betas_limbs,
                        'pose_rot6d': pred_pose_rot6d,       # used for pose prior...
                        'dog_breed': pred_breed,
                        'shapedirs': shapedirs_sel,
                        'z': pred_z,
                        'flength_unnorm': pred_flength,
                        'flength': output['flength'],
                        'partseg_images_rend': partseg_images,
                        'partseg_images_hg_nograd': partseg_images_hg,
                        'normflow_z': output['normflow_z']}

        return output, output_unnorm, output_reproj, output_ref_unnorm, output_orig_ref_comparison


    def forward_with_multiple_refinements(self, input_img, norm_dict=None, bone_lengths_prepared=None, betas=None):
        
        # run normal network part
        output, output_unnorm, output_reproj, output_ref_unnorm, output_orig_ref_comparison = self.forward(input_img, norm_dict=norm_dict, bone_lengths_prepared=bone_lengths_prepared, betas=betas)

        # prepare input for second refinement stage
        batch_size = output['keypoints_norm'].shape[0]
        keypoints_norm = output['keypoints_norm']
        pred_keyp_norm = (output_ref_unnorm['keyp_2d'].detach() / (self.image_size - 1) - 0.5)*2

        last_seg = output['seg_hg']
        pred_silh_images = output_ref_unnorm['silh'].detach() 

        trans_notnorm = output_orig_ref_comparison['ref_trans_notnorm']
        flength_notnorm = output_orig_ref_comparison['ref_flength_notnorm']
        # trans_notnorm = output_orig_ref_comparison['ref_pose_rotmat']
        pred_pose = output_ref_unnorm['pose_rotmat'].reshape((batch_size, -1, 3, 3))

        # run second refinement step
        output_ref_new = self.refinement_model(keypoints_norm.detach(), pred_keyp_norm, \
                            seg_sh_raw=last_seg[:, :, :, :].detach(), seg_pred_prep=pred_silh_images[:, :, :, :].detach()-0.5, \
                            in_pose_3x3=pred_pose.detach(), in_trans_notnorm=trans_notnorm.detach(), in_cam_notnorm=flength_notnorm.detach(), \
                            in_betas=output_ref_unnorm['betas'].detach(), in_betas_limbs=output_ref_unnorm['betas_limbs'].detach()) 
        # output_ref_new = self.refinement_model(keypoints_norm.detach(), pred_keyp_norm, seg_sh_raw=last_seg[:, :, :, :].detach(), seg_pred_prep=pred_silh_images[:, :, :, :].detach()-0.5, in_pose_3x3=pred_pose.detach(), in_trans_notnorm=trans_notnorm.detach(), in_cam_notnorm=flength_notnorm.detach(), in_betas=output_ref_unnorm['betas'].detach(), in_betas_limbs=output_ref_unnorm['betas_limbs'].detach()) 


        # new shape
        if self.shaperef_type == 'inexistent':
            if self.ref_detach_shape:
                output_ref_new['betas'] = output_ref_unnorm['betas'].detach()
                output_ref_new['betas_limbs'] = output_ref_unnorm['betas_limbs'].detach()
                output_ref_new['z'] = output_ref_unnorm['z'].detach()
                output_ref_new['shapedirs'] = output_reproj['shapedirs'].detach()
            else:
                output_ref_new['betas'] = output_ref_unnorm['betas']
                output_ref_new['betas_limbs'] = output_ref_unnorm['betas_limbs']
                output_ref_new['z'] = output_ref_unnorm['z']
                output_ref_new['shapedirs'] = output_reproj['shapedirs']
        else:
            assert ('betas' in output_ref_new.keys())
            assert ('betas_limbs' in output_ref_new.keys())
            output_ref_new['shapedirs'] = output_reproj['shapedirs']    

        # we denormalize flength and trans, but pose is handled differently
        if self.fix_flength:
            raise ValueError    # not sure if we want to have a fixed flength in refinement
        else:
            ref_pred_flength_orig = output_ref_new['flength_notnorm'] * norm_dict['flength_std'][None, :] + norm_dict['flength_mean'][None, :]   # (bs, 1)
            ref_pred_flength = ref_pred_flength_orig.clone()  # torch.abs(pred_flength_orig)
            ref_pred_flength[ref_pred_flength_orig<=0] = norm_dict['flength_mean'][None, :]
        ref_pred_trans = output_ref_new['trans_notnorm'] * norm_dict['trans_std'][None, :] + norm_dict['trans_mean'][None, :]    # (bs, 3)


        ref_pred_pose_reshx33 = output_ref_new['pose_rotmat'].reshape((batch_size, -1, 3, 3))
        ref_pred_pose_rot6d = rotmat_to_rot6d(ref_pred_pose_reshx33.reshape((-1, 3, 3))).reshape((batch_size, -1, 6))

        ref_V, ref_keyp_green_3d, _ = self.smal(beta=output_ref_new['betas'], betas_limbs=output_ref_new['betas_limbs'], 
                                        pose=ref_pred_pose_reshx33, trans=ref_pred_trans, get_skin=True, keyp_conf=self.smal_keyp_conf, 
                                        shapedirs_sel=output_ref_new['shapedirs'])

        ref_keyp_3d = ref_keyp_green_3d[:, :self.n_keyp, :]     # (bs, 20, 3)

        if not self.silh_no_tail:
            faces_prep = self.smal.faces.unsqueeze(0).expand((batch_size, -1, -1))
            ref_pred_silh_images, ref_pred_keyp = self.silh_renderer(vertices=ref_V, 
                points=ref_keyp_3d, faces=faces_prep, focal_lengths=ref_pred_flength)
        else:
            faces_no_tail_prep = torch.tensor(self.f_no_tail_np).to(device).expand((batch_size, -1, -1))
            ref_pred_silh_images, ref_pred_keyp = self.silh_renderer(vertices=ref_V, 
                points=ref_keyp_3d, faces=faces_no_tail_prep, focal_lengths=ref_pred_flength)

        output_ref_unnorm_new = {'vertices_smal': ref_V,
                            'keyp_3d': ref_keyp_3d,
                            'keyp_2d': ref_pred_keyp,
                            'silh': ref_pred_silh_images,
                            'trans': ref_pred_trans,
                            'flength': ref_pred_flength,
                            'betas': output_ref_new['betas'],
                            'betas_limbs': output_ref_new['betas_limbs'],
                            'pose_rot6d': ref_pred_pose_rot6d,   
                            'pose_rotmat':  ref_pred_pose_reshx33} 

        if not self.graphcnn_type == 'inexistent':
            output_ref_unnorm_new['vertexwise_ground_contact'] = output_ref_new['vertexwise_ground_contact']
        if not self.isflat_type=='inexistent':
            output_ref_unnorm_new['isflat'] = output_ref_new['isflat']
        if self.shaperef_type == 'inexistent':
            output_ref_unnorm_new['z'] = output_ref_new['z']

        output_orig_ref_comparison_new = {'ref_pose_rotmat': ref_pred_pose_reshx33,
                                    'ref_trans_notnorm': output_ref_new['trans_notnorm'],
                                    'ref_flength_notnorm': output_ref_new['flength_notnorm']}

        results = {
            'output': output, 
            'output_unnorm': output_unnorm, 
            'output_reproj':output_reproj, 
            'output_ref_unnorm': output_ref_unnorm, 
            'output_orig_ref_comparison':output_orig_ref_comparison,
            'output_ref_unnorm_new': output_ref_unnorm_new, 
            'output_orig_ref_comparison_new': output_orig_ref_comparison_new}
        return results


    def render_vis_nograd(self, vertices, focal_lengths, color=0):
        # this function is for visualization only
        # vertices: (bs, n_verts, 3)
        # focal_lengths: (bs, 1)
        # color: integer, either 0 or 1
        # returns a torch tensor of shape (bs, image_size, image_size, 3)
        with torch.no_grad():
            batch_size = vertices.shape[0]
            faces_prep = self.smal.faces.unsqueeze(0).expand((batch_size, -1, -1))
            visualizations = self.silh_renderer.get_visualization_nograd(vertices, 
                faces_prep, focal_lengths, color=color)
        return visualizations

