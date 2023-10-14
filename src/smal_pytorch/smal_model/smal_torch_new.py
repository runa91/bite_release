"""
PyTorch implementation of the SMAL/SMPL model
see:
    1.) https://github.com/silviazuffi/smalst/blob/master/smal_model/smal_torch.py
    2.) https://github.com/benjiebob/SMALify/blob/master/smal_model/smal_torch.py
main changes compared to SMALST and WLDO:
    * new model
        (...../side_packages/SMALify/new_smal_pca/results/my_tposeref_results_3/)
        dogs are part of the pca to create the model
        al meshes are centered around their root joint
        the animals are all scaled such that their body length (butt to breast) is 1
            X_init = np.concatenate((vertices_dogs, vertices_smal), axis=0)     # vertices_dogs
            X = []
            for ind in range(0, X_init.shape[0]):
                X_tmp, _, _, _ = align_smal_template_to_symmetry_axis(X_init[ind, :, :], subtract_mean=True)   # not sure if this is necessary
                X.append(X_tmp)
            X = np.asarray(X)
            # define points which will be used for normalization
            idxs_front = [6, 16, 8, 964]      #  [1172, 6, 16, 8, 964]
            idxs_back = [174, 2148, 175, 2149]       # not in the middle, but pairs
            reg_j = np.asarray(dd['J_regressor'].todense())
            # normalize the meshes such that X_frontback_dist is 1 and the root joint is in the center (0, 0, 0)
            X_front = X[:, idxs_front, :].mean(axis=1)
            X_back = X[:, idxs_back, :].mean(axis=1)
            X_frontback_dist = np.sqrt(((X_front - X_back)**2).sum(axis=1))
            X = X / X_frontback_dist[:, None, None]
            X_j0 = np.sum(X[:, reg_j[0, :]>0, :] * reg_j[0, (reg_j[0, :]>0)][None, :, None], axis=1)
            X = X - X_j0[:, None, :]
    * add limb length changes the same way as in WLDO
    * overall scale factor is added
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import chumpy as ch
import os.path
from torch import nn
from torch.autograd import Variable
import pickle as pkl 
from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation, batch_global_rigid_transformation_biggs, get_bone_length_scales, get_beta_scale_mask

from .smal_basics import align_smal_template_to_symmetry_axis, get_symmetry_indices

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.SMAL_configs import KEY_VIDS, CANONICAL_MODEL_JOINTS, CANONICAL_MODEL_JOINTS_REFINED, IDXS_BONES_NO_REDUNDANCY # , SMAL_MODEL_PATH
# from configs.SMAL_configs import SMAL_MODEL_TYPE
from configs.SMAL_configs import SMAL_MODEL_CONFIG

from smal_pytorch.utils import load_vertex_colors


# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r

# class SMAL(object):
class SMAL(nn.Module):        
    def __init__(self, pkl_path=None, smal_model_type=None, n_betas=None, template_name='neutral', use_smal_betas=True, logscale_part_list=None):
        super(SMAL, self).__init__()

        # before: pkl_path=SMAL_MODEL_PATH
        if smal_model_type is not None:
            assert (pkl_path is None)
            assert smal_model_type in SMAL_MODEL_CONFIG.keys()
            pkl_path = SMAL_MODEL_CONFIG[smal_model_type]['smal_model_path']
            self.smal_model_type = smal_model_type
            if logscale_part_list is None:
                logscale_part_list = SMAL_MODEL_CONFIG[smal_model_type]['logscale_part_list']
        elif (pkl_path is not None):
            self.smal_model_type = None
        elif (pkl_path is None):
            smal_model_type = 'barc'
            print('use default smal_model_type: ' + smal_model_type)
            pkl_path = SMAL_MODEL_CONFIG[smal_model_type]['smal_model_path']
            self.smal_model_type = smal_model_type
        else:
            raise ValueError


        '''
        # save some information about the model if possible
        if pkl_path == SMAL_MODEL_PATH:
            self.smal_model_type = SMAL_MODEL_TYPE
        '''

        if logscale_part_list is None:
            # logscale_part_list = ['front_legs_l', 'front_legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l', 'back_legs_l', 'back_legs_f'] 
            self.logscale_part_list = ['legs_l', 'legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l'] 
        else:
            self.logscale_part_list = logscale_part_list
        self.betas_scale_mask = get_beta_scale_mask(part_list=self.logscale_part_list)
        self.num_betas_logscale = len(self.logscale_part_list)

        self.use_smal_betas = use_smal_betas

        # -- Load SMPL params --
        try:
            with open(pkl_path, 'r') as f:
                dd = pkl.load(f)
        except (UnicodeDecodeError, TypeError) as e:
            with open(pkl_path, 'rb') as file:
                u = pkl._Unpickler(file)
                u.encoding = 'latin1'
                dd = u.load()

        self.f = dd['f']
        self.register_buffer('faces', torch.from_numpy(self.f.astype(int)))

        # get the correct template (mean shape)
        if template_name=='neutral':
            v_template = dd['v_template'] 
            v = v_template
        else:
            raise NotImplementedError

        # Mean template vertices
        self.register_buffer('v_template', torch.Tensor(v))
        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # symmetry indices
        self.sym_ids_dict = get_symmetry_indices()

        # Shape blend shape basis
        shapedir = np.reshape(undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        shapedir.flags['WRITEABLE'] = True      # not sure why this is necessary
        self.register_buffer('shapedirs', torch.Tensor(shapedir))

        # Regressor for joint locations given shape 
        self.register_buffer('J_regressor', torch.Tensor(dd['J_regressor'].T.todense()))

        # Pose blend shape basis
        num_pose_basis = dd['posedirs'].shape[-1]
        
        posedirs = np.reshape(undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.Tensor(posedirs))
        
        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self.register_buffer('weights', torch.Tensor(undo_chumpy(dd['weights'])))

        # prepare for vertex offsets
        self._prepare_for_vertex_offsets()


    def _prepare_for_vertex_offsets(self):
        sym_left_ids = self.sym_ids_dict['left']
        sym_right_ids = self.sym_ids_dict['right']
        sym_center_ids = self.sym_ids_dict['center']
        self.n_center = sym_center_ids.shape[0]
        self.n_left = sym_left_ids.shape[0]
        self.sl = 2*self.n_center       # sl: start left
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
        # self.register_buffer('inds_back_torch', torch.Tensor(inds_back).long())
        self.inds_back_torch = torch.Tensor(inds_back).long()
        return


    def _caclulate_bone_lengths_from_J(self, J, betas_logscale):
        # NEW: calculate bone lengths:
        all_bone_lengths_list = []
        for i in range(1, self.parents.shape[0]):
            bone_vec = J[:, i] - J[:, self.parents[i]]
            bone_length = torch.sqrt(torch.sum(bone_vec ** 2, axis=1))
            all_bone_lengths_list.append(bone_length)
        all_bone_lengths = torch.stack(all_bone_lengths_list)
        # some bones are pairs, it is enough to take one of the two bones  
        all_bone_length_scales = get_bone_length_scales(self.logscale_part_list, betas_logscale)
        all_bone_lengths = all_bone_lengths.permute((1,0)) * all_bone_length_scales

        return all_bone_lengths     #.permute((1,0))      


    def caclulate_bone_lengths(self, beta, betas_logscale, shapedirs_sel=None, short=True):
        nBetas = beta.shape[1]

        # 1. Add shape blend shapes
        # do we use the original shapedirs or a new set of selected shapedirs?
        if shapedirs_sel is None:
            shapedirs_sel = self.shapedirs[:nBetas,:]
        else: 
            assert shapedirs_sel.shape[0] == nBetas
        v_shaped = self.v_template + torch.reshape(torch.matmul(beta, shapedirs_sel), [-1, self.size[0], self.size[1]])

        # 2. Infer shape-dependent joint locations.
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        # calculate bone lengths
        all_bone_lengths = self._caclulate_bone_lengths_from_J(J, betas_logscale)
        selected_bone_lengths = all_bone_lengths[:, IDXS_BONES_NO_REDUNDANCY]
        
        if short:
            return selected_bone_lengths
        else:
            return all_bone_lengths



    def __call__(self, beta, betas_limbs, theta=None, pose=None, trans=None, del_v=None, get_skin=True, keyp_conf='red', get_all_info=False, shapedirs_sel=None, vert_off_compact=None):
        device = beta.device

        betas_logscale = betas_limbs
        # NEW: allow that rotation is given as rotation matrices instead of axis angle rotation
        #   theta: BSxNJointsx3 or BSx(NJoints*3)
        #   pose: NxNJointsx3x3
        if (theta is None) and (pose is None):
            raise ValueError("Either pose (rotation matrices NxNJointsx3x3) or theta (axis angle BSxNJointsx3) must be given")
        elif (theta is not None) and (pose is not None):
            raise ValueError("Not both pose (rotation matrices NxNJointsx3x3) and theta (axis angle BSxNJointsx3) can be given")

        if True:        # self.use_smal_betas:
            nBetas = beta.shape[1]
        else:
            nBetas = 0

        # add possibility to have additional vertex offsets
        if vert_off_compact is None:
            vertex_offsets = torch.zeros_like(self.v_template)
        else:
            if type(vert_off_compact) is dict:
                zero_vec = torch.zeros((vert_off_compact['c0'].shape[0], self.n_center)).to(device)
                half_vertex_offsets_center = torch.stack((vert_off_compact['c0'], \
                                                    zero_vec, \
                                                    vert_off_compact['c2']), axis=1)
                half_vertex_offsets_left = torch.stack((vert_off_compact['l0'], \
                                                    vert_off_compact['l1'], \
                                                    vert_off_compact['l2']), axis=1)
                half_vertex_offsets_right = torch.stack((vert_off_compact['l0'], \
                                                    - vert_off_compact['l1'], \
                                                    vert_off_compact['l2']), axis=1)
            else:
                zero_vec = torch.zeros((vert_off_compact.shape[0], self.n_center)).to(device)
                half_vertex_offsets_center = torch.stack((vert_off_compact[:, :self.n_center], \
                                                    zero_vec, \
                                                    vert_off_compact[:, self.n_center:2*self.n_center]), axis=1)
                half_vertex_offsets_left = torch.stack((vert_off_compact[:, self.sl:self.sl+self.n_left], \
                                                    vert_off_compact[:, self.sl+self.n_left:self.sl+2*self.n_left], \
                                                    vert_off_compact[:, self.sl+2*self.n_left:self.sl+3*self.n_left]), axis=1)
                half_vertex_offsets_right = torch.stack((vert_off_compact[:, self.sl:self.sl+self.n_left], \
                                                    - vert_off_compact[:, self.sl+self.n_left:self.sl+2*self.n_left], \
                                                    vert_off_compact[:, self.sl+2*self.n_left:self.sl+3*self.n_left]), axis=1)

            half_vertex_offsets_tot = torch.cat((half_vertex_offsets_center, half_vertex_offsets_left, half_vertex_offsets_right), dim=2)       # (bs, 3, 3889)
            vertex_offsets = torch.index_select(half_vertex_offsets_tot, dim=2, index=self.inds_back_torch.to(half_vertex_offsets_tot.device)).permute((0, 2, 1))     # (bs, 3889, 3)


        # 1. Add shape blend shapes
        # do we use the original shapedirs or a new set of selected shapedirs?
        if shapedirs_sel is None:
            shapedirs_sel = self.shapedirs[:nBetas,:]
        else: 
            assert shapedirs_sel.shape[0] == nBetas
        
        if nBetas > 0:
            if del_v is None:
                v_shaped = self.v_template + torch.reshape(torch.matmul(beta, shapedirs_sel), [-1, self.size[0], self.size[1]]) + vertex_offsets
            else:
                v_shaped = self.v_template + del_v + torch.reshape(torch.matmul(beta, shapedirs_sel), [-1, self.size[0], self.size[1]]) + vertex_offsets
        else:
            if del_v is None:
                v_shaped = self.v_template.unsqueeze(0) + vertex_offsets
            else:
                v_shaped = self.v_template + del_v + vertex_offsets

        # 2. Infer shape-dependent joint locations.
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        # 3. Add pose blend shapes
        # N x 24 x 3 x 3
        if pose is None:
            Rs = torch.reshape( batch_rodrigues(torch.reshape(theta, [-1, 3])), [-1, 35, 3, 3])
        else:
            Rs = pose
        # Ignore global rotation.
        pose_feature = torch.reshape(Rs[:, 1:, :, :] - torch.eye(3).to(device=device), [-1, 306])

        v_posed = torch.reshape(
            torch.matmul(pose_feature, self.posedirs),
            [-1, self.size[0], self.size[1]]) + v_shaped

        #-------------------------
        # new: add corrections of bone lengths to the template  (before hypothetical pose blend shapes!)
        # see biggs batch_lbs.py
        betas_scale = torch.exp(betas_logscale @ self.betas_scale_mask.to(betas_logscale.device))
        scaling_factors = betas_scale.reshape(-1, 35, 3)
        scale_factors_3x3 = torch.diag_embed(scaling_factors, dim1=-2, dim2=-1)

        # 4. Get the global joint location
        # self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents)
        self.J_transformed, A = batch_global_rigid_transformation_biggs(Rs, J, self.parents, scale_factors_3x3, betas_logscale=betas_logscale)

        # 2-BONES. Calculate bone lengths
        all_bone_lengths = self._caclulate_bone_lengths_from_J(J, betas_logscale)
        # selected_bone_lengths = all_bone_lengths[:, IDXS_BONES_NO_REDUNDANCY]
        #-------------------------

        # 5. Do skinning:
        num_batch = Rs.shape[0]
        
        weights_t = self.weights.repeat([num_batch, 1])
        W = torch.reshape(weights_t, [num_batch, -1, 35])

            
        T = torch.reshape(
            torch.matmul(W, torch.reshape(A, [num_batch, 35, 16])),
                [num_batch, -1, 4, 4])
        v_posed_homo = torch.cat(
                [v_posed, torch.ones([num_batch, v_posed.shape[1], 1]).to(device=device)], 2)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))

        verts = v_homo[:, :, :3, 0]

        if trans is None:
            trans = torch.zeros((num_batch,3)).to(device=device)

        verts = verts + trans[:,None,:]

        # Get joints:
        joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        # New... (see https://github.com/benjiebob/SMALify/blob/master/smal_model/smal_torch.py)
        joints = torch.cat([
            joints,
            verts[:, None, 1863],   # end_of_nose
            verts[:, None, 26],     # chin
            verts[:, None, 2124],   # right ear tip
            verts[:, None, 150],    # left ear tip
            verts[:, None, 3055],   # left eye
            verts[:, None, 1097],   # right eye
            # new: add paw keypoints, not joint locations -> bottom, rather in front
            # remark: when i look in the animals direction, left and right are exchanged
            verts[:, None, 1330],   # front paw, right
            verts[:, None, 3282],   # front paw, left
            verts[:, None, 1521],   # back paw, right
            verts[:, None, 3473],   # back paw, left
            verts[:, None, 6],      # throat
            verts[:, None, 20],     # withers
            ], dim = 1) 

        if keyp_conf == 'blue' or keyp_conf == 'dict':
            # Generate keypoints
            nLandmarks = KEY_VIDS.shape[0]      # 24
            j3d = torch.zeros((num_batch, nLandmarks, 3)).to(device=device)
            for j in range(nLandmarks):
                j3d[:, j,:] = torch.mean(verts[:, KEY_VIDS[j],:], dim=1)  # translation is already added to the vertices
            joints_blue = j3d

        joints_red = joints[:, :-12, :]  # joints[:, :-6, :]
        joints_green = joints[:, CANONICAL_MODEL_JOINTS, :]
        joints_olive = joints[:, CANONICAL_MODEL_JOINTS_REFINED, :]     # same order but better paw, withers and throat keypoints

        if keyp_conf == 'red':
            relevant_joints = joints_red
        elif keyp_conf == 'green':
            relevant_joints = joints_green
        elif keyp_conf == 'olive':
            relevant_joints = joints_olive
        elif keyp_conf == 'blue':
            relevant_joints = joints_blue
        elif keyp_conf == 'dict':
            relevant_joints = {'red': joints_red,
                            'green': joints_green,
                            'olive': joints_olive,
                            'blue': joints_blue}
        else:
            raise NotImplementedError

        if get_all_info:
            return verts, relevant_joints, Rs, all_bone_lengths
        else:
            if get_skin:
                return verts, relevant_joints, Rs        # , v_shaped
            else:
                return relevant_joints


    def get_joints_from_verts(self, verts, keyp_conf='red'):

        num_batch = verts.shape[0]

        # Get joints:
        joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        # New... (see https://github.com/benjiebob/SMALify/blob/master/smal_model/smal_torch.py)
        joints = torch.cat([
            joints,
            verts[:, None, 1863],   # end_of_nose
            verts[:, None, 26],     # chin
            verts[:, None, 2124],   # right ear tip
            verts[:, None, 150],    # left ear tip
            verts[:, None, 3055],   # left eye
            verts[:, None, 1097],   # right eye
            # new: add paw keypoints, not joint locations -> bottom, rather in front
            # remark: when i look in the animals direction, left and right are exchanged
            verts[:, None, 1330],   # front paw, right
            verts[:, None, 3282],   # front paw, left
            verts[:, None, 1521],   # back paw, right
            verts[:, None, 3473],   # back paw, left
            verts[:, None, 6],      # throat
            verts[:, None, 20],     # withers
            ], dim = 1) 


        if keyp_conf == 'blue' or keyp_conf == 'dict':
            # Generate keypoints
            nLandmarks = KEY_VIDS.shape[0]      # 24
            j3d = torch.zeros((num_batch, nLandmarks, 3)).to(device=device)
            for j in range(nLandmarks):
                j3d[:, j,:] = torch.mean(verts[:, KEY_VIDS[j],:], dim=1)  # translation is already added to the vertices
            joints_blue = j3d

        joints_red = joints[:, :-12, :]  # joints[:, :-6, :]
        joints_green = joints[:, CANONICAL_MODEL_JOINTS, :]
        joints_olive = joints[:, CANONICAL_MODEL_JOINTS_REFINED, :]     # same order but better paw, withers and throat keypoints

        if keyp_conf == 'red':
            relevant_joints = joints_red
        elif keyp_conf == 'green':
            relevant_joints = joints_green
        elif keyp_conf == 'olive':
            relevant_joints = joints_olive
        elif keyp_conf == 'blue':
            relevant_joints = joints_blue
        elif keyp_conf == 'dict':
            relevant_joints = {'red': joints_red,
                            'green': joints_green,
                            'olive': joints_olive,
                            'blue': joints_blue}

        return relevant_joints




