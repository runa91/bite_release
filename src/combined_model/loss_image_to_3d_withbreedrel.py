

import torch
import numpy as np
import pickle as pkl

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from priors.normalizing_flow_prior.normalizing_flow_prior import NormalizingFlowPrior
from priors.shape_prior import ShapePrior
from lifting_to_3d.utils.geometry_utils import rot6d_to_rotmat, batch_rot2aa
from configs.SMAL_configs import SMAL_MODEL_CONFIG

from priors.helper_3dcgmodel_loss import load_dog_betas_for_3dcgmodel_loss
from combined_model.loss_utils.loss_utils_gc import calculate_plane_errors_batch



class Loss(torch.nn.Module):
    def __init__(self, smal_model_type, data_info, nf_version=None):
        super(Loss, self).__init__()
        self.criterion_regr = torch.nn.MSELoss()        # takes the mean   
        self.criterion_class = torch.nn.CrossEntropyLoss()
        self.data_info = data_info   
        self.register_buffer('keypoint_weights', torch.tensor(data_info.keypoint_weights)[None, :])
        self.l_anchor = None
        self.l_pos = None
        self.l_neg = None
        self.smal_model_type = smal_model_type
        self.smal_model_data_path = SMAL_MODEL_CONFIG[self.smal_model_type]['smal_model_data_path']
        self.unity_smal_shape_prior_dogs = SMAL_MODEL_CONFIG[self.smal_model_type]['unity_smal_shape_prior_dogs']

        if nf_version is not None:
            self.normalizing_flow_pose_prior = NormalizingFlowPrior(nf_version=nf_version)
        self.shape_prior = ShapePrior(self.smal_model_data_path) # here we just need mean and cov
        self.criterion_triplet = torch.nn.TripletMarginLoss(margin=1)

        # load 3d data for the unity dogs (an optional shape prior for 11 breeds)
        if self.unity_smal_shape_prior_dogs is not None:
            self.dog_betas_unity = load_dog_betas_for_3dcgmodel_loss(self.unity_smal_shape_prior_dogs, self.smal_model_type)
        else:
            self.dog_betas_unity = None

        root_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')  
        remeshing_path = os.path.join(root_data_path, 'smal_data_remeshed', 'uniform_surface_sampling', 'my_smpl_39dogsnorm_Jr_4_dog_remesh4000_info.pkl')
        with open(remeshing_path, 'rb') as fp: 
            self.remeshing_dict = pkl.load(fp)
        self.remeshing_relevant_faces = torch.tensor(self.remeshing_dict['smal_faces'][self.remeshing_dict['faceid_closest']], dtype=torch.long)
        self.remeshing_relevant_barys = torch.tensor(self.remeshing_dict['barys_closest'], dtype=torch.float32)


    def prepare_anchor_pos_neg(self, batch_size, device):
        l0 = np.arange(0, batch_size, 2)
        l_anchor = []
        l_pos = []
        l_neg = []
        for ind in l0:
            xx = set(np.arange(0, batch_size))
            xx.discard(ind)
            xx.discard(ind+1)
            for ind2 in xx:
                if ind2 % 2 == 0:
                    l_anchor.append(ind)
                    l_pos.append(ind + 1)
                else:
                    l_anchor.append(ind + 1)
                    l_pos.append(ind)
                l_neg.append(ind2)
        self.l_anchor = torch.Tensor(l_anchor).to(torch.int64).to(device)
        self.l_pos = torch.Tensor(l_pos).to(torch.int64).to(device)
        self.l_neg = torch.Tensor(l_neg).to(torch.int64).to(device)
        return


    def forward(self, output_reproj, target_dict, weight_dict=None):

        # output_reproj: ['vertices_smal', 'keyp_3d', 'keyp_2d', 'silh_image']
        # target_dict: ['index', 'center', 'scale', 'pts', 'tpts', 'target_weight']
        batch_size = output_reproj['keyp_2d'].shape[0]
        device = output_reproj['keyp_2d'].device

        # loss on reprojected keypoints 
        output_kp_resh = (output_reproj['keyp_2d']).reshape((-1, 2))    
        target_kp_resh = (target_dict['tpts'][:, :, :2] / 64. * (256. - 1)).reshape((-1, 2))
        weights_resh = target_dict['tpts'][:, :, 2].reshape((-1)) 
        keyp_w_resh = self.keypoint_weights.repeat((batch_size, 1)).reshape((-1))
        loss_keyp = ((((output_kp_resh - target_kp_resh)[weights_resh>0]**2).sum(axis=1).sqrt()*weights_resh[weights_resh>0])*keyp_w_resh[weights_resh>0]).sum() / \
            max((weights_resh[weights_resh>0]*keyp_w_resh[weights_resh>0]).sum(), 1e-5)

        # loss on reprojected silhouette
        assert output_reproj['silh'].shape == (target_dict['silh'][:, None, :, :]).shape
        silh_loss_type = 'default'
        if silh_loss_type == 'default':
            with torch.no_grad():
                thr_silh = 20
                diff = torch.norm(output_kp_resh - target_kp_resh, dim=1)
                diff_x = diff.reshape((batch_size, -1))
                weights_resh_x = weights_resh.reshape((batch_size, -1))
                unweighted_kp_mean_dist = (diff_x * weights_resh_x).sum(dim=1) / ((weights_resh_x).sum(dim=1)+1e-6)
            loss_silh_bs = ((output_reproj['silh'] - target_dict['silh'][:, None, :, :]) ** 2).sum(axis=3).sum(axis=2).sum(axis=1) / (output_reproj['silh'].shape[2]*output_reproj['silh'].shape[3])
            loss_silh = loss_silh_bs[unweighted_kp_mean_dist<thr_silh].sum() / batch_size
        else:
            print('silh_loss_type: ' + silh_loss_type)
            raise ValueError

        # shape regularization
        #   'smal': loss on betas (pca coefficients), betas should be close to 0
        #   'limbs...' loss on selected betas_limbs
        loss_shape_weighted_list = [torch.zeros((1), device=device).mean().to(output_reproj['keyp_2d'].device)]  
        for ind_sp, sp in enumerate(weight_dict['shape_options']):
            weight_sp = weight_dict['shape'][ind_sp]
            # self.logscale_part_list = ['legs_l', 'legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l'] 
            if sp == 'smal':
                loss_shape_tmp = self.shape_prior(output_reproj['betas'])
            elif sp == 'limbs':
                loss_shape_tmp = torch.mean((output_reproj['betas_limbs'])**2)  
            elif sp == 'limbs7':
                limb_coeffs_list = [0.01, 1, 0.1, 1, 1, 0.1, 2]
                limb_coeffs = torch.tensor(limb_coeffs_list).to(torch.float32).to(target_dict['tpts'].device)   
                loss_shape_tmp = torch.mean((output_reproj['betas_limbs'] * limb_coeffs[None, :])**2)            
            else:
                raise NotImplementedError
            loss_shape_weighted_list.append(weight_sp * loss_shape_tmp)
        loss_shape_weighted = torch.stack((loss_shape_weighted_list)).sum()

        # 3D loss for dogs for which we have a unity model or toy figure
        loss_models3d = torch.zeros((1), device=device).mean().to(output_reproj['betas'].device)
        if 'models3d' in weight_dict.keys():
            if weight_dict['models3d'] > 0:
                assert (self.dog_betas_unity is not None)
                if weight_dict['models3d'] > 0:
                    for ind_dog in range(target_dict['breed_index'].shape[0]):
                        breed_index = np.asscalar(target_dict['breed_index'][ind_dog].detach().cpu().numpy())
                        if breed_index in self.dog_betas_unity.keys():
                            betas_target = self.dog_betas_unity[breed_index][:output_reproj['betas'].shape[1]].to(output_reproj['betas'].device)
                            betas_output = output_reproj['betas'][ind_dog, :]
                            betas_limbs_output = output_reproj['betas_limbs'][ind_dog, :]
                            loss_models3d += ((betas_limbs_output**2).sum() + ((betas_output-betas_target)**2).sum()) / (output_reproj['betas'].shape[1] + output_reproj['betas_limbs'].shape[1])
            else:
                weight_dict['models3d'] = 0.0
        else:
            weight_dict['models3d'] = 0.0

        # shape resularization loss on shapedirs
        #   -> in the current version shapedirs are kept fixed, so we don't need those losses
        if weight_dict['shapedirs'] > 0:
            raise NotImplementedError  
        else:
            loss_shapedirs = torch.zeros((1), device=device).mean().to(output_reproj['betas'].device)

        # prior on back joints (not used in cvpr 2022 paper)
        #   -> elementwise MSE loss on all 6 coefficients of 6d rotation representation
        if 'pose_0' in weight_dict.keys(): 
            if weight_dict['pose_0'] > 0:
                pred_pose_rot6d = output_reproj['pose_rot6d']
                w_rj_np = np.zeros((pred_pose_rot6d.shape[1]))
                w_rj_np[[2, 3, 4, 5]] = 1.0         # back
                w_rj = torch.tensor(w_rj_np).to(torch.float32).to(pred_pose_rot6d.device)     
                zero_rot = torch.tensor([1, 0, 0, 1, 0, 0]).to(pred_pose_rot6d.device).to(torch.float32)[None, None, :].repeat((batch_size, pred_pose_rot6d.shape[1], 1))
                loss_pose = self.criterion_regr(pred_pose_rot6d*w_rj[None, :, None], zero_rot*w_rj[None, :, None])
            else:
                loss_pose = torch.zeros((1), device=device).mean()

        # pose prior 
        #   -> we did experiment with different pose priors, for example:
        #       * similart to SMALify (https://github.com/benjiebob/SMALify/blob/master/smal_fitter/smal_fitter.py, 
        #         https://github.com/benjiebob/SMALify/blob/master/smal_fitter/priors/pose_prior_35.py)
        #       * vae 
        #       * normalizing flow pose prior
        #   -> our cvpr 2022 paper uses the normalizing flow pose prior as implemented below
        if 'poseprior' in weight_dict.keys():
            if weight_dict['poseprior'] > 0:
                pred_pose_rot6d = output_reproj['pose_rot6d']
                pred_pose = rot6d_to_rotmat(pred_pose_rot6d.reshape((-1, 6))).reshape((batch_size, -1, 3, 3))
                if 'normalizing_flow_tiger' in weight_dict['poseprior_options']:
                    if output_reproj['normflow_z'] is not None:
                        loss_poseprior = self.normalizing_flow_pose_prior.calculate_loss_from_z(output_reproj['normflow_z'], type='square')
                    else:
                        loss_poseprior = self.normalizing_flow_pose_prior.calculate_loss(pred_pose_rot6d, type='square')
                elif 'normalizing_flow_tiger_logprob' in weight_dict['poseprior_options']:
                    if output_reproj['normflow_z'] is not None:
                        loss_poseprior = self.normalizing_flow_pose_prior.calculate_loss_from_z(output_reproj['normflow_z'], type='neg_log_prob')
                    else:
                        loss_poseprior = self.normalizing_flow_pose_prior.calculate_loss(pred_pose_rot6d, type='neg_log_prob')
                else:
                    raise NotImplementedError
            else:
                loss_poseprior = torch.zeros((1), device=device).mean()
        else:
            weight_dict['poseprior'] = 0
            loss_poseprior = torch.zeros((1), device=device).mean()

        # add a prior which penalizes side-movement angles for legs 
        if 'poselegssidemovement' in weight_dict.keys():
            if weight_dict['poselegssidemovement'] > 0:
                use_pose_legs_side_loss = True
            else:
                use_pose_legs_side_loss = False
        else:
            use_pose_legs_side_loss = False
        if use_pose_legs_side_loss:
            leg_indices_right = np.asarray([7, 8, 9, 10, 17, 18, 19, 20])      # front, back
            leg_indices_left = np.asarray([11, 12, 13, 14, 21, 22, 23, 24])     # front, back
            vec = torch.zeros((3, 1)).to(device=pred_pose.device, dtype=pred_pose.dtype)
            vec[2] = -1
            x0_rotmat = pred_pose   
            x0_rotmat_legs_left = x0_rotmat[:, leg_indices_left, :, :]
            x0_rotmat_legs_right = x0_rotmat[:, leg_indices_right, :, :]
            x0_legs_left = x0_rotmat_legs_left.reshape((-1, 3, 3))@vec
            x0_legs_right = x0_rotmat_legs_right.reshape((-1, 3, 3))@vec
            eps=0       # 1e-7
            # use the component of the vector which points to the side
            loss_poselegssidemovement = (x0_legs_left[:, 1]**2).mean() + (x0_legs_right[:, 1]**2).mean()
        else:
            loss_poselegssidemovement = torch.zeros((1), device=device).mean()
            weight_dict['poselegssidemovement'] = 0

        # dog breed classification loss
        dog_breed_gt = target_dict['breed_index']
        dog_breed_pred = output_reproj['dog_breed']
        loss_class = self.criterion_class(dog_breed_pred, dog_breed_gt)

        # dog breed relationship loss
        #   -> we did experiment with many other options, but none was significantly better 
        if '4' in weight_dict['breed_options']:      # we have pairs of dogs of the same breed 
            if weight_dict['breed'] > 0:
                assert output_reproj['dog_breed'].shape[0] == 12
                # assert weight_dict['breed'] > 0
                z = output_reproj['z']   
                # go through all pairs and compare them to each other sample
                if self.l_anchor is None:
                    self.prepare_anchor_pos_neg(batch_size, z.device)
                anchor = torch.index_select(z, 0, self.l_anchor)
                positive = torch.index_select(z, 0, self.l_pos)
                negative = torch.index_select(z, 0, self.l_neg)
                loss_breed = self.criterion_triplet(anchor, positive, negative)
            else:
                loss_breed = torch.zeros((1), device=device).mean()
        else:
            loss_breed = torch.zeros((1), device=device).mean()

        # regularizarion for focal length
        loss_flength_near_mean = torch.mean(output_reproj['flength']**2)
        loss_flength = loss_flength_near_mean

        # bodypart segmentation loss
        if 'partseg' in weight_dict.keys():
            if weight_dict['partseg'] > 0:
                raise NotImplementedError
            else:
                loss_partseg = torch.zeros((1), device=device).mean()
        else:
            weight_dict['partseg'] = 0
            loss_partseg = torch.zeros((1), device=device).mean()


        # NEW: ground contact loss for main network
        keep_smal_mesh = False 
        if 'gc_plane' in weight_dict.keys():
            if weight_dict['gc_plane'] > 0:
                if keep_smal_mesh:
                    target_gc_class = target_dict['gc'][:, :, 0]
                    gc_errors_plane = calculate_plane_errors_batch(output_reproj['vertices_smal'], target_gc_class, target_dict['has_gc'], target_dict['has_gc_is_touching'])
                    loss_gc_plane = torch.mean(gc_errors_plane)
                else:   # use a uniformly sampled mesh
                    target_gc_class = target_dict['gc'][:, :, 0]
                    device = output_reproj['vertices_smal'].device
                    remeshing_relevant_faces = self.remeshing_relevant_faces.to(device)
                    remeshing_relevant_barys = self.remeshing_relevant_barys.to(device)

                    bs = output_reproj['vertices_smal'].shape[0]
                    # verts_remeshed = torch.einsum('ij,aijk->aik', remeshing_relevant_barys, output_reproj['vertices_smal'][:, self.remeshing_relevant_faces])
                    # sel_verts_comparison = output_reproj['vertices_smal'][:, self.remeshing_relevant_faces]
                    # verts_remeshed = torch.einsum('ij,aijk->aik', remeshing_relevant_barys, sel_verts_comparison)
                    sel_verts = torch.index_select(output_reproj['vertices_smal'], dim=1, index=remeshing_relevant_faces.reshape((-1))).reshape((bs, remeshing_relevant_faces.shape[0], 3, 3))
                    verts_remeshed = torch.einsum('ij,aijk->aik', remeshing_relevant_barys, sel_verts)
                    target_gc_class_remeshed = torch.einsum('ij,aij->ai', remeshing_relevant_barys, target_gc_class[:, self.remeshing_relevant_faces].to(device=device, dtype=torch.float32))
                    target_gc_class_remeshed_prep = torch.round(target_gc_class_remeshed).to(torch.long)
                    gc_errors_plane, gc_errors_under_plane = calculate_plane_errors_batch(verts_remeshed, target_gc_class_remeshed_prep, target_dict['has_gc'], target_dict['has_gc_is_touching'])
                    loss_gc_plane = torch.mean(gc_errors_plane) 
                    loss_gc_belowplane = torch.mean(gc_errors_under_plane)
                    # loss_dict_temp['gc_plane'] = torch.mean(gc_errors_plane)
            else:
                loss_gc_plane = torch.zeros((1), device=device).mean()
                loss_gc_belowplane = torch.zeros((1), device=device).mean()
        else:
            loss_gc_plane = torch.zeros((1), device=device).mean()
            loss_gc_belowplane = torch.zeros((1), device=device).mean()
            weight_dict['gc_plane'] = 0
            weight_dict['gc_belowplane'] = 0



        # weight and combine losses
        loss_keyp_weighted = loss_keyp * weight_dict['keyp']
        loss_silh_weighted = loss_silh * weight_dict['silh']
        loss_shapedirs_weighted = loss_shapedirs * weight_dict['shapedirs']
        loss_pose_weighted = loss_pose * weight_dict['pose_0']
        loss_class_weighted = loss_class * weight_dict['class']
        loss_breed_weighted = loss_breed * weight_dict['breed']
        loss_flength_weighted = loss_flength * weight_dict['flength']
        loss_poseprior_weighted = loss_poseprior * weight_dict['poseprior']
        loss_partseg_weighted = loss_partseg * weight_dict['partseg']
        loss_models3d_weighted = loss_models3d * weight_dict['models3d']
        loss_poselegssidemovement_weighted = loss_poselegssidemovement * weight_dict['poselegssidemovement']

        loss_gc_plane_weighted = loss_gc_plane * weight_dict['gc_plane']
        loss_gc_belowplane_weighted = loss_gc_belowplane * weight_dict['gc_belowplane']


        ####################################################################################################
        loss = loss_keyp_weighted + loss_silh_weighted + loss_shape_weighted + loss_pose_weighted + loss_class_weighted + \
                loss_shapedirs_weighted + loss_breed_weighted + loss_flength_weighted + loss_poseprior_weighted + \
                loss_partseg_weighted + loss_models3d_weighted + loss_poselegssidemovement_weighted + \
                loss_gc_plane_weighted + loss_gc_belowplane_weighted
        ####################################################################################################
        
        loss_dict = {'loss': loss.item(), 
                    'loss_keyp_weighted': loss_keyp_weighted.item(), \
                    'loss_silh_weighted': loss_silh_weighted.item(), \
                    'loss_shape_weighted': loss_shape_weighted.item(), \
                    'loss_shapedirs_weighted': loss_shapedirs_weighted.item(), \
                    'loss_pose0_weighted': loss_pose_weighted.item(), \
                    'loss_class_weighted': loss_class_weighted.item(), \
                    'loss_breed_weighted': loss_breed_weighted.item(), \
                    'loss_flength_weighted': loss_flength_weighted.item(), \
                    'loss_poseprior_weighted': loss_poseprior_weighted.item(), \
                    'loss_partseg_weighted': loss_partseg_weighted.item(), \
                    'loss_models3d_weighted': loss_models3d_weighted.item(), \
                    'loss_poselegssidemovement_weighted': loss_poselegssidemovement_weighted.item(), \
                    'loss_gc_plane_weighted': loss_gc_plane_weighted.item(), \
                    'loss_gc_belowplane_weighted': loss_gc_belowplane_weighted.item()
                    }
                    
        return loss, loss_dict




