

import torch
import numpy as np
import pickle as pkl

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from priors.normalizing_flow_prior.normalizing_flow_prior import NormalizingFlowPrior
from priors.shape_prior import ShapePrior
from lifting_to_3d.utils.geometry_utils import rot6d_to_rotmat, batch_rot2aa, geodesic_loss_R
from combined_model.loss_utils.loss_utils import leg_sideway_error, leg_torsion_error, tail_sideway_error, tail_torsion_error, spine_torsion_error, spine_sideway_error
from combined_model.loss_utils.loss_utils_gc import LossGConMesh, calculate_plane_errors_batch

from priors.shape_prior import ShapePrior
from configs.SMAL_configs import SMAL_MODEL_CONFIG

from priors.helper_3dcgmodel_loss import load_dog_betas_for_3dcgmodel_loss


class LossRef(torch.nn.Module):
    def __init__(self, smal_model_type, data_info, nf_version=None):
        super(LossRef, self).__init__()
        self.criterion_regr = torch.nn.MSELoss()        # takes the mean   
        self.criterion_class = torch.nn.CrossEntropyLoss()

        class_weights_isflat = torch.tensor([12, 2])
        self.criterion_class_isflat = torch.nn.CrossEntropyLoss(weight=class_weights_isflat)
        self.criterion_l1 = torch.nn.L1Loss()
        self.geodesic_loss = geodesic_loss_R(reduction='mean')
        self.gc_loss_on_mesh = LossGConMesh()
        self.data_info = data_info   
        self.smal_model_type = smal_model_type
        self.register_buffer('keypoint_weights', torch.tensor(data_info.keypoint_weights)[None, :])
        # if nf_version is not None:
        #     self.normalizing_flow_pose_prior = NormalizingFlowPrior(nf_version=nf_version)

        self.smal_model_data_path = SMAL_MODEL_CONFIG[self.smal_model_type]['smal_model_data_path']
        self.shape_prior = ShapePrior(self.smal_model_data_path) # here we just need mean and cov        

        # remeshing as used for ground contact
        root_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')  
        remeshing_path = os.path.join(root_data_path, 'smal_data_remeshed', 'uniform_surface_sampling', 'my_smpl_39dogsnorm_Jr_4_dog_remesh4000_info.pkl')
        with open(remeshing_path, 'rb') as fp: 
            self.remeshing_dict = pkl.load(fp)
        self.remeshing_relevant_faces = torch.tensor(self.remeshing_dict['smal_faces'][self.remeshing_dict['faceid_closest']], dtype=torch.long)
        self.remeshing_relevant_barys = torch.tensor(self.remeshing_dict['barys_closest'], dtype=torch.float32)

        # load 3d data for the unity dogs (an optional shape prior for 11 breeds)
        self.unity_smal_shape_prior_dogs = SMAL_MODEL_CONFIG[self.smal_model_type]['unity_smal_shape_prior_dogs']
        if self.unity_smal_shape_prior_dogs is not None:
            self.dog_betas_unity = load_dog_betas_for_3dcgmodel_loss(self.unity_smal_shape_prior_dogs, self.smal_model_type)
        else:
            self.dog_betas_unity = None


    def forward(self, output_ref, output_ref_comp, target_dict, weight_dict_ref):
        # output_reproj: ['vertices_smal', 'keyp_3d', 'keyp_2d', 'silh_image']
        # target_dict: ['index', 'center', 'scale', 'pts', 'tpts', 'target_weight']
        batch_size = output_ref['keyp_2d'].shape[0]
        loss_dict_temp = {}

        # loss on reprojected keypoints 
        output_kp_resh = (output_ref['keyp_2d']).reshape((-1, 2))    
        target_kp_resh = (target_dict['tpts'][:, :, :2] / 64. * (256. - 1)).reshape((-1, 2))
        weights_resh = target_dict['tpts'][:, :, 2].reshape((-1)) 
        keyp_w_resh = self.keypoint_weights.repeat((batch_size, 1)).reshape((-1))
        loss_dict_temp['keyp_ref'] = ((((output_kp_resh - target_kp_resh)[weights_resh>0]**2).sum(axis=1).sqrt()*weights_resh[weights_resh>0])*keyp_w_resh[weights_resh>0]).sum() / \
            max((weights_resh[weights_resh>0]*keyp_w_resh[weights_resh>0]).sum(), 1e-5)

        # loss on reprojected silhouette
        assert output_ref['silh'].shape == (target_dict['silh'][:, None, :, :]).shape
        silh_loss_type = 'default'
        if silh_loss_type == 'default':
            with torch.no_grad():
                thr_silh = 20
                diff = torch.norm(output_kp_resh - target_kp_resh, dim=1)
                diff_x = diff.reshape((batch_size, -1))
                weights_resh_x = weights_resh.reshape((batch_size, -1))
                unweighted_kp_mean_dist = (diff_x * weights_resh_x).sum(dim=1) / ((weights_resh_x).sum(dim=1)+1e-6)
            loss_silh_bs = ((output_ref['silh'] - target_dict['silh'][:, None, :, :]) ** 2).sum(axis=3).sum(axis=2).sum(axis=1) / (output_ref['silh'].shape[2]*output_ref['silh'].shape[3])
            loss_dict_temp['silh_ref'] = loss_silh_bs[unweighted_kp_mean_dist<thr_silh].sum() / batch_size
        else:
            print('silh_loss_type: ' + silh_loss_type)
            raise ValueError

        # regularization: losses on difference between previous prediction and refinement
        loss_dict_temp['reg_trans'] = self.criterion_l1(output_ref_comp['ref_trans_notnorm'], output_ref_comp['old_trans_notnorm'].detach()) * 3
        loss_dict_temp['reg_flength'] = self.criterion_l1(output_ref_comp['ref_flength_notnorm'], output_ref_comp['old_flength_notnorm'].detach()) * 1
        loss_dict_temp['reg_pose'] = self.geodesic_loss(output_ref_comp['ref_pose_rotmat'], output_ref_comp['old_pose_rotmat'].detach()) * 35 * 6

        # pose priors on refined pose
        loss_dict_temp['pose_legs_side'] = leg_sideway_error(output_ref['pose_rotmat'])
        loss_dict_temp['pose_legs_tors'] = leg_torsion_error(output_ref['pose_rotmat'])
        loss_dict_temp['pose_tail_side'] = tail_sideway_error(output_ref['pose_rotmat'])
        loss_dict_temp['pose_tail_tors'] = tail_torsion_error(output_ref['pose_rotmat'])
        loss_dict_temp['pose_spine_side'] = spine_sideway_error(output_ref['pose_rotmat'])
        loss_dict_temp['pose_spine_tors'] = spine_torsion_error(output_ref['pose_rotmat'])

        # loss to predict ground contact per vertex
        # import pdb; pdb.set_trace()
        if 'gc_vertexwise' in weight_dict_ref.keys():
            # import pdb; pdb.set_trace()
            device = output_ref['vertexwise_ground_contact'].device
            pred_gc = output_ref['vertexwise_ground_contact']
            loss_dict_temp['gc_vertexwise'] = self.gc_loss_on_mesh(pred_gc, target_dict['gc'].to(device=device, dtype=torch.long), target_dict['has_gc'], loss_type_gcmesh='ce')

        keep_smal_mesh = False 
        if 'gc_plane' in weight_dict_ref.keys():
            if weight_dict_ref['gc_plane'] > 0:
                if keep_smal_mesh:
                    target_gc_class = target_dict['gc'][:, :, 0]
                    gc_errors_plane = calculate_plane_errors_batch(output_ref['vertices_smal'], target_gc_class, target_dict['has_gc'], target_dict['has_gc_is_touching'])
                    loss_dict_temp['gc_plane'] = torch.mean(gc_errors_plane)
                else:   # use a uniformly sampled mesh
                    target_gc_class = target_dict['gc'][:, :, 0]
                    device = output_ref['vertices_smal'].device
                    remeshing_relevant_faces = self.remeshing_relevant_faces.to(device)
                    remeshing_relevant_barys = self.remeshing_relevant_barys.to(device)

                    bs = output_ref['vertices_smal'].shape[0]
                    # verts_remeshed = torch.einsum('ij,aijk->aik', remeshing_relevant_barys, output_ref['vertices_smal'][:, self.remeshing_relevant_faces])
                    # sel_verts_comparison = output_ref['vertices_smal'][:, self.remeshing_relevant_faces]
                    # verts_remeshed = torch.einsum('ij,aijk->aik', remeshing_relevant_barys, sel_verts_comparison)
                    sel_verts = torch.index_select(output_ref['vertices_smal'], dim=1, index=remeshing_relevant_faces.reshape((-1))).reshape((bs, remeshing_relevant_faces.shape[0], 3, 3))
                    verts_remeshed = torch.einsum('ij,aijk->aik', remeshing_relevant_barys, sel_verts)
                    target_gc_class_remeshed = torch.einsum('ij,aij->ai', remeshing_relevant_barys, target_gc_class[:, self.remeshing_relevant_faces].to(device=device, dtype=torch.float32))
                    target_gc_class_remeshed_prep = torch.round(target_gc_class_remeshed).to(torch.long)
                    gc_errors_plane, gc_errors_under_plane = calculate_plane_errors_batch(verts_remeshed, target_gc_class_remeshed_prep, target_dict['has_gc'], target_dict['has_gc_is_touching'])
                    loss_dict_temp['gc_plane'] = torch.mean(gc_errors_plane) 
                    loss_dict_temp['gc_blowplane'] = torch.mean(gc_errors_under_plane)

        # error on classification if the ground plane is flat
        if 'gc_isflat' in weight_dict_ref.keys():
            # import pdb; pdb.set_trace()
            self.criterion_class_isflat.to(device)
            loss_dict_temp['gc_isflat'] = self.criterion_class(output_ref['isflat'], target_dict['isflat'].to(device))

        # if we refine the shape WITHIN the refinement newtork (shaperef_type is not inexistent)
        # shape regularization
        #   'smal': loss on betas (pca coefficients), betas should be close to 0
        #   'limbs...' loss on selected betas_limbs
        device = output_ref_comp['ref_trans_notnorm'].device
        loss_shape_weighted_list = [torch.zeros((1), device=device).mean()]  
        if 'shape_options' in weight_dict_ref.keys():
            for ind_sp, sp in enumerate(weight_dict_ref['shape_options']):
                weight_sp = weight_dict_ref['shape'][ind_sp]
                # self.logscale_part_list = ['legs_l', 'legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l'] 
                if sp == 'smal':
                    loss_shape_tmp = self.shape_prior(output_ref['betas'])
                elif sp == 'limbs':
                    loss_shape_tmp = torch.mean((output_ref['betas_limbs'])**2)  
                elif sp == 'limbs7':
                    limb_coeffs_list = [0.01, 1, 0.1, 1, 1, 0.1, 2]
                    limb_coeffs = torch.tensor(limb_coeffs_list).to(torch.float32).to(target_dict['tpts'].device)   
                    loss_shape_tmp = torch.mean((output_ref['betas_limbs'] * limb_coeffs[None, :])**2)            
                else:
                    raise NotImplementedError
                loss_shape_weighted_list.append(weight_sp * loss_shape_tmp)
        loss_shape_weighted = torch.stack((loss_shape_weighted_list)).sum()

        # 3D loss for dogs for which we have a unity model or toy figure
        loss_dict_temp['models3d'] = torch.zeros((1), device=device).mean().to(output_ref['betas'].device)
        if 'models3d' in weight_dict_ref.keys():
            if weight_dict_ref['models3d'] > 0:
                assert (self.dog_betas_unity is not None)
                if weight_dict_ref['models3d'] > 0:
                    for ind_dog in range(target_dict['breed_index'].shape[0]):
                        breed_index = np.asscalar(target_dict['breed_index'][ind_dog].detach().cpu().numpy())
                        if breed_index in self.dog_betas_unity.keys():
                            betas_target = self.dog_betas_unity[breed_index][:output_ref['betas'].shape[1]].to(output_ref['betas'].device)
                            betas_output = output_ref['betas'][ind_dog, :]
                            betas_limbs_output = output_ref['betas_limbs'][ind_dog, :]
                            loss_dict_temp['models3d'] += ((betas_limbs_output**2).sum() + ((betas_output-betas_target)**2).sum()) / (output_ref['betas'].shape[1] + output_ref['betas_limbs'].shape[1])
            else:
                weight_dict_ref['models3d'] = 0.0
        else:
            weight_dict_ref['models3d'] = 0.0

        # weight the losses
        loss = torch.zeros((1)).mean().to(device=output_ref['keyp_2d'].device, dtype=output_ref['keyp_2d'].dtype)
        loss_dict = {}
        for loss_name in weight_dict_ref.keys():
            if not loss_name in ['shape', 'shape_options']:
                if weight_dict_ref[loss_name] > 0:
                    loss_weighted = loss_dict_temp[loss_name] * weight_dict_ref[loss_name]
                    loss_dict[loss_name] = loss_weighted.item()
                    loss += loss_weighted
        loss += loss_shape_weighted
        loss_dict['loss'] = loss.item()

        return loss, loss_dict


