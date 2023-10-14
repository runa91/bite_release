
# aenv_new_icon_2

# was used for ttoptv6_sketchfab_v16: python src/test_time_optimization/ttopt_fromref_v6_sketchfab.py --workers 12 --save-images True --config refinement_cfg_visualization_withgc_withvertexwisegc_isflat.yaml --model-file-complete=cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0/checkpoint.pth.tar --sketchfab 1

# for stanext images:
#   python scripts/ttopt_fromref_v7_sketchfab.py --workers 12 --config refinement_cfg_test_withvertexwisegc_csaddnonflat.yaml --model-file-complete=cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0/checkpoint.pth.tar -s ttopt_vtest1
# for all images from the folder datasets/test_image_crops:
#   python scripts/ttopt_fromref_v7_sketchfab.py --workers 12 --config refinement_cfg_test_withvertexwisegc_csaddnonflat_crops.yaml --model-file-complete=cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0/checkpoint.pth.tar -s ttopt_vtest2



import argparse
import os.path
import json
import numpy as np
import pickle as pkl
import csv
from distutils.util import strtobool
import torch
from torch import nn
import torch.backends.cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from collections import OrderedDict
import glob
from tqdm import tqdm
from dominate import document
from dominate.tags import *
from PIL import Image
from matplotlib import pyplot as plt
import trimesh
import cv2
import shutil

from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency


import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from combined_model.train_main_image_to_3d_wbr_withref import do_validation_epoch
from combined_model.model_shape_v7_withref_withgraphcnn import ModelImageTo3d_withshape_withproj 

# from combined_model.loss_image_to_3d_withbreedrel import Loss
# from combined_model.loss_image_to_3d_refinement import LossRef
from configs.barc_cfg_defaults import get_cfg_defaults, update_cfg_global_with_yaml, get_cfg_global_updated

from lifting_to_3d.utils.geometry_utils import rot6d_to_rotmat, rotmat_to_rot6d  
from stacked_hourglass.datasets.utils_dataset_selection import get_evaluation_dataset, get_sketchfab_evaluation_dataset, get_crop_evaluation_dataset, get_norm_dict

from test_time_optimization.bite_inference_model_for_ttopt import BITEInferenceModel
from smal_pytorch.smal_model.smal_torch_new import SMAL
from configs.SMAL_configs import SMAL_MODEL_CONFIG
from smal_pytorch.renderer.differentiable_renderer import SilhRenderer
from test_time_optimization.utils.utils_ttopt import reset_loss_values, get_optimed_pose_with_glob

from combined_model.loss_utils.loss_utils import leg_sideway_error, leg_torsion_error, tail_sideway_error, tail_torsion_error, spine_torsion_error, spine_sideway_error
from combined_model.loss_utils.loss_utils_gc import LossGConMesh, calculate_plane_errors_batch
from combined_model.loss_utils.loss_arap import Arap_Loss
from combined_model.loss_utils.loss_laplacian_mesh_comparison import LaplacianCTF     # (coarse to fine animal)
from graph_networks import graphcmr     # .utils_mesh import Mesh
from stacked_hourglass.utils.visualization import save_input_image_with_keypoints, save_input_image

# from src.evaluation.sketchfab_evaluation.alignment_utils.calculate_v2v_error_release import compute_similarity_transform
# from src.evaluation.sketchfab_evaluation.alignment_utils.calculate_alignment_error import calculate_alignemnt_errors





def main(args):

    # load configs
    #   step 1: load default configs
    #   step 2: load updates from .yaml file
    path_config = os.path.join(get_cfg_defaults().barc_dir, 'src', 'configs', args.config)
    update_cfg_global_with_yaml(path_config)
    cfg = get_cfg_global_updated()

    # define path to load the trained model
    path_model_file_complete = os.path.join(cfg.paths.ROOT_CHECKPOINT_PATH, args.model_file_complete) 

    # define and create paths to save results
    out_sub_name = cfg.data.VAL_OPT + '_' + cfg.data.DATASET + '_' + args.suffix + '/'
    root_out_path = os.path.join(os.path.dirname(path_model_file_complete).replace(cfg.paths.ROOT_CHECKPOINT_PATH, cfg.paths.ROOT_OUT_PATH + 'results_ttopt/'), out_sub_name)
    root_out_path_details = root_out_path + 'details/'
    if not os.path.exists(root_out_path): os.makedirs(root_out_path)
    if not os.path.exists(root_out_path_details): os.makedirs(root_out_path_details)
    print('root_out_path: ' + root_out_path)

    # other paths
    root_data_path = os.path.join(os.path.dirname(__file__), '../', 'data')
    # downsampling as used in graph neural network
    root_smal_downsampling = os.path.join(root_data_path, 'graphcmr_data')   
    # remeshing as used for ground contact
    remeshing_path = os.path.join(root_data_path, 'smal_data_remeshed', 'uniform_surface_sampling', 'my_smpl_39dogsnorm_Jr_4_dog_remesh4000_info.pkl')

    loss_weight_path = os.path.join(os.path.dirname(__file__), '../', 'src', 'configs', 'ttopt_loss_weights', args.loss_weight_ttopt_path)  
    print(loss_weight_path)
    with open(loss_weight_path, 'r') as j: 
        losses = json.loads(j.read())
    shutil.copyfile(loss_weight_path, root_out_path_details + os.path.basename(loss_weight_path))

    # Select the hardware device to use for training.
    if torch.cuda.is_available() and cfg.device=='cuda':
        device = torch.device('cuda', torch.cuda.current_device())
        torch.backends.cudnn.benchmark = False      # True
    else:
        device = torch.device('cpu')

    print('structure_pose_net: ' + cfg.params.STRUCTURE_POSE_NET)
    print('refinement network type: ' + cfg.params.REF_NET_TYPE)
    print('smal_model_type: ' + cfg.smal.SMAL_MODEL_TYPE)

    print(losses)

    # prepare dataset and dataset loader
    if cfg.data.DATASET == 'sketchfab':
        print('load sketchfab dataset')
        val_dataset, val_loader, len_val_dataset, test_name_list, stanext_data_info, stanext_acc_joints = get_sketchfab_evaluation_dataset(cfg.optim.BATCH_SIZE, args.workers)
    elif cfg.data.DATASET == 'ImgCropList': 
        input_folder = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'test_image_crops')
        val_dataset, val_loader, len_val_dataset, test_name_list, stanext_data_info, stanext_acc_joints = get_crop_evaluation_dataset(cfg.optim.BATCH_SIZE, args.workers, input_folder)
    elif cfg.data.DATASET in ['stanext24_easy', 'stanext24', 'stanext24_withgc', 'stanext24_withgc_big']:
        print('load dataset')
        val_dataset, val_loader, len_val_dataset, test_name_list, stanext_data_info, stanext_acc_joints = get_evaluation_dataset(cfg.data.DATASET, cfg.data.VAL_OPT, cfg.data.V12, cfg.optim.BATCH_SIZE, args.workers)
    else:
        raise NotImplementedError

    # summarize information for normalization 
    norm_dict = get_norm_dict(stanext_data_info, device)

    # prepare complete model
    bite_model = BITEInferenceModel(cfg, path_model_file_complete, norm_dict)
    smal_model_type = bite_model.smal_model_type
    logscale_part_list = SMAL_MODEL_CONFIG[smal_model_type]['logscale_part_list']       # ['legs_l', 'legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l'] 
    smal = SMAL(smal_model_type=smal_model_type, template_name='neutral', logscale_part_list=logscale_part_list).to(device)    
    silh_renderer = SilhRenderer(image_size=256).to(device)    
    keypoint_weights = torch.tensor(stanext_data_info.keypoint_weights, dtype=torch.float)[None, :].to(device) 

    # load loss modules -> not necessary!
    # loss_module = Loss(smal_model_type=cfg.smal.SMAL_MODEL_TYPE, data_info=StanExt.DATA_INFO, nf_version=cfg.params.NF_VERSION).to(device)    
    # loss_module_ref = LossRef(smal_model_type=cfg.smal.SMAL_MODEL_TYPE, data_info=StanExt.DATA_INFO, nf_version=cfg.params.NF_VERSION).to(device)    

    # remeshing utils
    with open(remeshing_path, 'rb') as fp: 
        remeshing_dict = pkl.load(fp)
    remeshing_relevant_faces = torch.tensor(remeshing_dict['smal_faces'][remeshing_dict['faceid_closest']], dtype=torch.long, device=device)
    remeshing_relevant_barys = torch.tensor(remeshing_dict['barys_closest'], dtype=torch.float32, device=device)

    # prepare progress bar
    iterable = enumerate(val_loader)
    progress = None
    if True:        # not quiet:
        progress = tqdm(iterable, desc='Train', total=len(val_loader), ascii=True, leave=False)
        iterable = progress
    ind_img_tot = 0

    for i, (input, target_dict) in iterable:
        batch_size = input.shape[0]
        # prepare variables, put them on the right device
        for key in target_dict.keys(): 
            if key == 'breed_index':
                target_dict[key] = target_dict[key].long().to(device)
            elif key in ['index', 'pts', 'tpts', 'target_weight', 'silh', 'silh_distmat_tofg', 'silh_distmat_tobg', 'sim_breed_index', 'img_border_mask']:
                target_dict[key] = target_dict[key].float().to(device)
            elif key == 'has_seg':
                target_dict[key] = target_dict[key].to(device)
            else:
                pass
        input = input.float().to(device)

        # get starting values for the optimization 
        preds_dict = bite_model.get_all_results(input)
        # res_normal_and_ref = bite_model.get_selected_results(preds_dict=preds_dict, result_networks=['normal', 'ref'])
        res = bite_model.get_selected_results(preds_dict=preds_dict, result_networks=['ref'])['ref']
        bs = res['pose_rotmat'].shape[0]
        all_pose_6d = rotmat_to_rot6d(res['pose_rotmat'][:, None, 1:, :, :].clone().reshape((-1, 3, 3))).reshape((bs, -1, 6))       # [bs, 34, 6]
        all_orient_6d = rotmat_to_rot6d(res['pose_rotmat'][:, None, :1, :, :].clone().reshape((-1, 3, 3))).reshape((bs, -1, 6))     # [bs, 1, 6]
        
        # --------------------------------------------------------------------
        for ind_img in range(input.shape[0]):
            name = (test_name_list[target_dict['index'][ind_img].long()]).replace('/', '__').split('.')[0]

            print('ind_img_tot: ' + str(ind_img_tot) + '   -> ' + name)
            ind_img_tot += 1
            batch_size = 1

            # save initial visualizations
            # save the image with keypoints as predicted by the stacked hourglass
            pred_unp_prep = torch.cat((res['hg_keyp_256'][ind_img, :, :].detach(), res['hg_keyp_scores'][ind_img, :, :]), 1)
            inp_img = input[ind_img, :, :, :].detach().clone()
            out_path = root_out_path +  name + '_hg_key.png'
            save_input_image_with_keypoints(inp_img, pred_unp_prep, out_path=out_path, threshold=0.01, print_scores=True, ratio_in_out=1.0)    # threshold=0.3
            # save the input image
            img_inp = input[ind_img, :, :, :].clone()
            for t, m, s in zip(img_inp, stanext_data_info.rgb_mean, stanext_data_info.rgb_stddev): t.add_(m)       # inverse to transforms.color_normalize()
            img_inp = img_inp.detach().cpu().numpy().transpose(1, 2, 0)  
            img_init = Image.fromarray(np.uint8(255*img_inp)).convert('RGB') 
            img_init.save(root_out_path_details + name + '_img_ainit.png')
            # save ground truth silhouette (for visualization only, it is not used during the optimization)
            target_img_silh = Image.fromarray(np.uint8(255*target_dict['silh'][ind_img, :, :].detach().cpu().numpy())).convert('RGB')
            target_img_silh.save(root_out_path_details +  name + '_target_silh.png')
            # save the silhouette as predicted by the stacked hourglass
            hg_img_silh = Image.fromarray(np.uint8(255*res['hg_silh_prep'][ind_img, :, :].detach().cpu().numpy())).convert('RGB')
            hg_img_silh.save(root_out_path +  name + '_hg_silh.png')

            # initialize the variables over which we want to optimize
            optimed_pose_6d = all_pose_6d[ind_img, None, :, :].to(device).clone().detach().requires_grad_(True)
            optimed_orient_6d = all_orient_6d[ind_img, None, :, :].to(device).clone().detach().requires_grad_(True)  # [1, 1, 6]
            optimed_betas = res['betas'][ind_img, None, :].to(device).clone().detach().requires_grad_(True)   # [1,30]
            optimed_trans_xy = res['trans'][ind_img, None, :2].to(device).clone().detach().requires_grad_(True)
            optimed_trans_z =res['trans'][ind_img, None, 2:3].to(device).clone().detach().requires_grad_(True)
            optimed_camera_flength = res['flength'][ind_img, None, :].to(device).clone().detach().requires_grad_(True)  # [1,1]
            n_vert_comp = 2*smal.n_center + 3*smal.n_left
            optimed_vert_off_compact = torch.tensor(np.zeros((batch_size, n_vert_comp)), dtype=torch.float,
                                        device=device,
                                        requires_grad=True) 
            assert len(logscale_part_list) == 7
            new_betas_limb_lengths = res['betas_limbs'][ind_img, None, :]
            optimed_betas_limbs = new_betas_limb_lengths.to(device).clone().detach().requires_grad_(True)  # [1,7]

            # define the optimizers 
            optimizer = torch.optim.SGD(
                # [optimed_pose, optimed_trans_xy, optimed_betas, optimed_betas_limbs, optimed_orient, optimed_vert_off_compact],
                [optimed_camera_flength, optimed_trans_z, optimed_trans_xy, optimed_pose_6d, optimed_orient_6d, optimed_betas, optimed_betas_limbs],
                lr=5*1e-4,        # 1e-3,
                momentum=0.9)
            optimizer_vshift = torch.optim.SGD(
                [optimed_camera_flength, optimed_trans_z, optimed_trans_xy, optimed_pose_6d, optimed_orient_6d, optimed_betas, optimed_betas_limbs, optimed_vert_off_compact],
                lr=1e-4,  # 1e-4,
                momentum=0.9)
            nopose_optimizer = torch.optim.SGD(
                # [optimed_pose, optimed_trans_xy, optimed_betas, optimed_betas_limbs, optimed_orient, optimed_vert_off_compact],
                [optimed_camera_flength, optimed_trans_z, optimed_trans_xy, optimed_orient_6d, optimed_betas, optimed_betas_limbs],
                lr=5*1e-4,        # 1e-3,
                momentum=0.9)
            nopose_optimizer_vshift = torch.optim.SGD(
                [optimed_camera_flength, optimed_trans_z, optimed_trans_xy, optimed_orient_6d, optimed_betas, optimed_betas_limbs, optimed_vert_off_compact],
                lr=1e-4,  # 1e-4,
                momentum=0.9)
            # define schedulers
            patience = 5
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                verbose=0,
                min_lr=1e-5,
                patience=patience)
            scheduler_vshift = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_vshift,
                mode='min',
                factor=0.5,
                verbose=0,
                min_lr=1e-5,
                patience=patience)

            # set all loss values to 0
            losses = reset_loss_values(losses)

            # prepare all the target labels: keypoints, silhouette, ground contact, ...
            with torch.no_grad():
                thr_kp = 0.2
                kp_weights = res['hg_keyp_scores']
                kp_weights[res['hg_keyp_scores']<thr_kp] = 0
                weights_resh = kp_weights[ind_img, None, :, :].reshape((-1))  # target_dict['tpts'][:, :, 2].reshape((-1)) 
                keyp_w_resh = keypoint_weights.repeat((batch_size, 1)).reshape((-1))
                # prepare predicted ground contact labels
                sm = nn.Softmax(dim=1)
                target_gc_class = sm(res['vertexwise_ground_contact'][ind_img, :, :])[None, :, 1]       # values between 0 and 1
                target_gc_class_remeshed = torch.einsum('ij,aij->ai', remeshing_relevant_barys, target_gc_class[:, remeshing_relevant_faces].to(device=device, dtype=torch.float32))
                target_gc_class_remeshed_prep = torch.round(target_gc_class_remeshed).to(torch.long)
                vert_colors = np.repeat(255*target_gc_class.detach().cpu().numpy()[0, :, None], 3, 1)
                vert_colors[:, 2] = 255
                faces_prep = smal.faces.unsqueeze(0).expand((batch_size, -1, -1))
                # prepare target silhouette and keypoints, from stacked hourglass predictions
                target_hg_silh = res['hg_silh_prep'][ind_img, :, :].detach()
                target_kp_resh = res['hg_keyp_256'][ind_img, None, :, :].reshape((-1, 2)).detach()
                # find out if ground contact constraints should be used for the image at hand
                # print('is flat: ' + str(res['isflat_prep'][ind_img]))
                if res['isflat_prep'][ind_img] >= 0.5: # threshold should probably be set higher
                    isflat = [True]
                else:
                    isflat = [False] 
                if target_gc_class_remeshed_prep.sum() > 3:
                    istouching = [True]
                else:
                    istouching = [False]
                ignore_pose_optimization = False


            ##########################################################################################################
            # start optimizing for this image
            n_iter = 301    # how many iterations are desired? (+1)
            loop = tqdm(range(n_iter))
            per_loop_lst = []
            list_error_procrustes = []
            for i in loop:
                # for the first 150 iterations steps we don't allow vertex shifts
                if i == 0:          
                    current_i = 0
                    if ignore_pose_optimization:
                        current_optimizer = nopose_optimizer
                    else:
                        current_optimizer = optimizer
                    current_scheduler = scheduler
                    current_weight_name = 'weight' 
                # after 150 iteration steps we start with vertex shifts    
                elif i == 150:      
                    current_i = 0
                    if ignore_pose_optimization:
                        current_optimizer = nopose_optimizer_vshift
                    else:
                        current_optimizer = optimizer_vshift
                    current_scheduler = scheduler_vshift
                    current_weight_name = 'weight_vshift'    
                    # set up arap loss
                    if losses["arap"]['weight_vshift'] > 0.0:
                        with torch.no_grad():
                            torch_mesh_comparison = Meshes(smal_verts.detach(), faces_prep.detach())
                        arap_loss = Arap_Loss(meshes=torch_mesh_comparison, device=device)  
                    #  is there a laplacian loss similar as in coarse-to-fine?
                    if losses["lapctf"]['weight_vshift'] > 0.0:
                        torch_verts_comparison = smal_verts.detach().clone()
                        smal_model_type_downsampling = '39dogs_norm'
                        smal_downsampling_npz_name = 'mesh_downsampling_' + os.path.basename(SMAL_MODEL_CONFIG[smal_model_type_downsampling]['smal_model_path']).replace('.pkl', '_template.npz')
                        smal_downsampling_npz_path = os.path.join(root_smal_downsampling, smal_downsampling_npz_name)  
                        data = np.load(smal_downsampling_npz_path, encoding='latin1', allow_pickle=True)
                        adjmat = data['A'][0]
                        laplacian_ctf = LaplacianCTF(adjmat, device=device)
                else:
                    pass


                current_optimizer.zero_grad()

                # get 3d smal model
                optimed_pose_with_glob = get_optimed_pose_with_glob(optimed_orient_6d, optimed_pose_6d)
                optimed_trans = torch.cat((optimed_trans_xy, optimed_trans_z), dim=1)
                smal_verts, keyp_3d, _ = smal(beta=optimed_betas, betas_limbs=optimed_betas_limbs, pose=optimed_pose_with_glob, vert_off_compact=optimed_vert_off_compact, trans=optimed_trans, keyp_conf='olive', get_skin=True)

                # render silhouette and keypoints
                pred_silh_images, pred_keyp_raw = silh_renderer(vertices=smal_verts, points=keyp_3d, faces=faces_prep, focal_lengths=optimed_camera_flength)
                pred_keyp = pred_keyp_raw[:, :24, :]

                # save silhouette reprojection visualization
                if i==0:
                    img_silh = Image.fromarray(np.uint8(255*pred_silh_images[0, 0, :, :].detach().cpu().numpy())).convert('RGB')
                    img_silh.save(root_out_path_details +  name + '_silh_ainit.png')
                    my_mesh_tri = trimesh.Trimesh(vertices=smal_verts[0, ...].detach().cpu().numpy(), faces=faces_prep[0, ...].detach().cpu().numpy(), process=False,  maintain_order=True)
                    my_mesh_tri.export(root_out_path_details +  name + '_res_ainit.obj')

                # silhouette loss
                diff_silh = torch.abs(pred_silh_images[0, 0, :, :] - target_hg_silh)
                losses['silhouette']['value'] = diff_silh.mean()

                # keypoint_loss
                output_kp_resh = (pred_keyp[0, :, :]).reshape((-1, 2))    
                losses['keyp']['value'] = ((((output_kp_resh - target_kp_resh)[weights_resh>0]**2).sum(axis=1).sqrt() * \
                    weights_resh[weights_resh>0])*keyp_w_resh[weights_resh>0]).sum() / \
                    max((weights_resh[weights_resh>0]*keyp_w_resh[weights_resh>0]).sum(), 1e-5)
                # losses['keyp']['value'] = ((((output_kp_resh - target_kp_resh)[weights_resh>0]**2).sum(axis=1).sqrt()*weights_resh[weights_resh>0])*keyp_w_resh[weights_resh>0]).sum() / max((weights_resh[weights_resh>0]*keyp_w_resh[weights_resh>0]).sum(), 1e-5)

                # pose priors on refined pose
                losses['pose_legs_side']['value'] = leg_sideway_error(optimed_pose_with_glob)
                losses['pose_legs_tors']['value'] = leg_torsion_error(optimed_pose_with_glob)
                losses['pose_tail_side']['value'] = tail_sideway_error(optimed_pose_with_glob)
                losses['pose_tail_tors']['value'] = tail_torsion_error(optimed_pose_with_glob)
                losses['pose_spine_side']['value'] = spine_sideway_error(optimed_pose_with_glob)
                losses['pose_spine_tors']['value'] = spine_torsion_error(optimed_pose_with_glob)

                # ground contact loss
                sel_verts = torch.index_select(smal_verts, dim=1, index=remeshing_relevant_faces.reshape((-1))).reshape((batch_size, remeshing_relevant_faces.shape[0], 3, 3))
                verts_remeshed = torch.einsum('ij,aijk->aik', remeshing_relevant_barys, sel_verts)

                # gc_errors_plane, gc_errors_under_plane = calculate_plane_errors_batch(verts_remeshed, target_gc_class_remeshed_prep, target_dict['has_gc'], target_dict['has_gc_is_touching'])
                gc_errors_plane, gc_errors_under_plane = calculate_plane_errors_batch(verts_remeshed, target_gc_class_remeshed_prep, isflat, istouching)

                losses['gc_plane']['value'] = torch.mean(gc_errors_plane) 
                losses['gc_belowplane']['value'] = torch.mean(gc_errors_under_plane)

                # edge length of the predicted mesh
                if (losses["edge"][current_weight_name] + losses["normal"][ current_weight_name] + losses["laplacian"][ current_weight_name]) > 0:
                    torch_mesh = Meshes(smal_verts, faces_prep.detach())
                    losses["edge"]['value'] = mesh_edge_loss(torch_mesh)
                    # mesh normal consistency
                    losses["normal"]['value'] = mesh_normal_consistency(torch_mesh)
                    # mesh laplacian smoothing
                    losses["laplacian"]['value'] = mesh_laplacian_smoothing(torch_mesh, method="uniform")

                # arap loss
                if losses["arap"][current_weight_name] > 0.0:
                    torch_mesh = Meshes(smal_verts, faces_prep.detach())
                    losses["arap"]['value'] =  arap_loss(torch_mesh)

                # laplacian loss for comparison (from coarse-to-fine paper)
                if losses["lapctf"][current_weight_name] > 0.0:
                    verts_refine = smal_verts
                    loss_almost_arap, loss_smooth = laplacian_ctf(verts_refine, torch_verts_comparison)
                    losses["lapctf"]['value'] =  loss_almost_arap

                # Weighted sum of the losses
                total_loss = 0.0 
                for k in ['keyp', 'silhouette', 'pose_legs_side', 'pose_legs_tors', 'pose_tail_side', 'pose_tail_tors', 'pose_spine_tors', 'pose_spine_side', 'gc_plane', 'gc_belowplane', 'edge', 'normal', 'laplacian', 'arap', 'lapctf']:
                    if losses[k][current_weight_name] > 0.0:
                        total_loss += losses[k]['value'] * losses[k][current_weight_name]

                # calculate gradient and make optimization step
                total_loss.backward(retain_graph=True)  #  
                current_optimizer.step()
                current_scheduler.step(total_loss)
                loop.set_description(f"Body Fitting = {total_loss.item():.3f}")

                # save the result three times (0, 150, 300)
                if i % 150 == 0:    
                    # save silhouette image
                    img_silh = Image.fromarray(np.uint8(255*pred_silh_images[0, 0, :, :].detach().cpu().numpy())).convert('RGB')
                    img_silh.save(root_out_path_details +  name + '_silh_e' + format(i, '03d') + '.png')
                    # save image overlay
                    visualizations = silh_renderer.get_visualization_nograd(smal_verts, faces_prep, optimed_camera_flength, color=0)
                    pred_tex = visualizations[0, :, :, :].permute((1, 2, 0)).cpu().detach().numpy() / 256
                    # out_path = root_out_path_details +  name + '_tex_pred_e' + format(i, '03d') + '.png'
                    # plt.imsave(out_path, pred_tex)
                    input_image_np = img_inp.copy()
                    im_masked = cv2.addWeighted(input_image_np,0.2,pred_tex,0.8,0)
                    pred_tex_max = np.max(pred_tex, axis=2)
                    im_masked[pred_tex_max<0.01, :] = input_image_np[pred_tex_max<0.01, :]
                    out_path = root_out_path +  name + '_comp_pred_e' + format(i, '03d') + '.png'
                    plt.imsave(out_path, im_masked)
                    # save mesh
                    my_mesh_tri = trimesh.Trimesh(vertices=smal_verts[0, ...].detach().cpu().numpy(), faces=faces_prep[0, ...].detach().cpu().numpy(), process=False,  maintain_order=True)
                    my_mesh_tri.visual.vertex_colors = vert_colors
                    my_mesh_tri.export(root_out_path +  name + '_res_e' + format(i, '03d') + '.obj')
                    # save focal length (together with the mesh this is enough to create an overlay in blender)
                    out_file_flength = root_out_path_details +  name + '_flength_e' + format(i, '03d') # + '.npz'
                    np.save(out_file_flength, optimed_camera_flength.detach().cpu().numpy())
                current_i += 1

            ##########################################################################################################








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a stacked hourglass model.')
    parser.add_argument('--model-file-complete', default='', type=str, metavar='PATH',
                        help='path to saved model weights')
    parser.add_argument('-lwttopt', '--loss-weight-ttopt-path', default='bite_loss_weights_ttopt.json', type=str, metavar='PATH',
                        help='name of json file which contains the loss weights')
    parser.add_argument('-cg', '--config', default='barc_cfg_test.yaml', type=str, metavar='PATH',
                        help='name of config file (default: barc_cfg_test.yaml within src/configs folder)')
    parser.add_argument('-s', '--suffix', default='ttopt_v0', type=str,
                        help='suffix for name of the result folder')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    main(parser.parse_args())