
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
import random
from datetime import datetime
import gradio as gr

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency


import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from combined_model.train_main_image_to_3d_wbr_withref import do_validation_epoch
from combined_model.model_shape_v7_withref_withgraphcnn import ModelImageTo3d_withshape_withproj 

from configs.barc_cfg_defaults import get_cfg_defaults, update_cfg_global_with_yaml, get_cfg_global_updated

from lifting_to_3d.utils.geometry_utils import rot6d_to_rotmat, rotmat_to_rot6d  
from stacked_hourglass.datasets.utils_dataset_selection import get_evaluation_dataset, get_sketchfab_evaluation_dataset, get_crop_evaluation_dataset, get_norm_dict, get_single_crop_dataset_from_image

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

random.seed(2)

print(
    "torch: ", torch.__version__,
    "\ntorchvision: ", torchvision.__version__,
)


def get_prediction(model, img_path_or_img, confidence=0.5):
    """
    see https://haochen23.github.io/2020/04/object-detection-faster-rcnn.html#.YsMCm4TP3-g
    get_prediction
        parameters:
        - img_path - path of the input image
        - confidence - threshold value for prediction score
        method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
            are chosen.
    """
    if isinstance(img_path_or_img, str):
        img = Image.open(img_path_or_img).convert('RGB')
    else:
        img = img_path_or_img
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    # pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_class = list(pred[0]['labels'].numpy())
    pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    try:
        pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        return pred_boxes, pred_class, pred_score
    except:
        print('no bounding box with a score that is high enough found! -> work on full image')
        return None, None, None


def detect_object(model, img_path_or_img, confidence=0.5, rect_th=2, text_size=0.5, text_th=1):
    """
    see https://haochen23.github.io/2020/04/object-detection-faster-rcnn.html#.YsMCm4TP3-g
    object_detection_api
        parameters:
        - img_path_or_img - path of the input image
        - confidence - threshold value for prediction score
        - rect_th - thickness of bounding box
        - text_size - size of the class label text
        - text_th - thichness of the text
        method:
        - prediction is obtained from get_prediction method
        - for each prediction, bounding box is drawn and text is written 
            with opencv
        - the final image is displayed
    """
    boxes, pred_cls, pred_scores = get_prediction(model, img_path_or_img, confidence)
    if isinstance(img_path_or_img, str):
        img = cv2.imread(img_path_or_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = img_path_or_img
    is_first = True
    bbox = None
    if boxes is not None:
        for i in range(len(boxes)):
            cls = pred_cls[i]
            if cls == 18 and bbox is None:
                cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
                # cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
                # cv2.putText(img, str(pred_scores[i]), boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
                bbox = boxes[i]
    return img, bbox


# -------------------------------------------------------------------------------------------------------------------- #
model_bbox = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_bbox.eval()

def run_bbox_inference(input_image):
    # load configs
    cfg = get_cfg_global_updated()
    out_path = os.path.join(cfg.paths.ROOT_OUT_PATH, 'gradio_examples', 'test2.png')
    img, bbox = detect_object(model=model_bbox, img_path_or_img=input_image, confidence=0.5)
    # fig = plt.figure()   #  plt.figure(figsize=(20,30))
    # plt.imsave(out_path, img)
    return img, bbox



# -------------------------------------------------------------------------------------------------------------------- #
args_config = "refinement_cfg_test_withvertexwisegc_csaddnonflat.yaml"
# args_model_file_complete = "cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0/checkpoint.pth.tar"
args_model_file_complete = "cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0_forrelease_v0/checkpoint.pth.tar"
args_suffix = "ttopt_v0"
args_loss_weight_ttopt_path = "bite_loss_weights_ttopt.json"
args_workers = 12
# -------------------------------------------------------------------------------------------------------------------- #



# load configs
#   step 1: load default configs
#   step 2: load updates from .yaml file
path_config = os.path.join(get_cfg_defaults().barc_dir, 'src', 'configs', args_config)
update_cfg_global_with_yaml(path_config)
cfg = get_cfg_global_updated()

# define path to load the trained model
path_model_file_complete = os.path.join(cfg.paths.ROOT_CHECKPOINT_PATH, args_model_file_complete) 

# define and create paths to save results
out_sub_name = cfg.data.VAL_OPT + '_' + cfg.data.DATASET + '_' + args_suffix + '/'
root_out_path = os.path.join(os.path.dirname(path_model_file_complete).replace(cfg.paths.ROOT_CHECKPOINT_PATH, cfg.paths.ROOT_OUT_PATH + 'results_gradio/'), out_sub_name)
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

loss_weight_path = os.path.join(os.path.dirname(__file__), '../', 'src', 'configs', 'ttopt_loss_weights', args_loss_weight_ttopt_path)  
print(loss_weight_path)


# Select the hardware device to use for training.
if torch.cuda.is_available() and cfg.device=='cuda':
    device = torch.device('cuda', torch.cuda.current_device())
    torch.backends.cudnn.benchmark = False      # True
else:
    device = torch.device('cpu')

print('structure_pose_net: ' + cfg.params.STRUCTURE_POSE_NET)
print('refinement network type: ' + cfg.params.REF_NET_TYPE)
print('smal_model_type: ' + cfg.smal.SMAL_MODEL_TYPE)

# prepare complete model
norm_dict = get_norm_dict(data_info=None, device=device)
bite_model = BITEInferenceModel(cfg, path_model_file_complete, norm_dict)
smal_model_type = bite_model.smal_model_type
logscale_part_list = SMAL_MODEL_CONFIG[smal_model_type]['logscale_part_list']       # ['legs_l', 'legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l'] 
smal = SMAL(smal_model_type=smal_model_type, template_name='neutral', logscale_part_list=logscale_part_list).to(device)    
silh_renderer = SilhRenderer(image_size=256).to(device)    

# load loss modules -> not necessary!
# loss_module = Loss(smal_model_type=cfg.smal.SMAL_MODEL_TYPE, data_info=StanExt.DATA_INFO, nf_version=cfg.params.NF_VERSION).to(device)    
# loss_module_ref = LossRef(smal_model_type=cfg.smal.SMAL_MODEL_TYPE, data_info=StanExt.DATA_INFO, nf_version=cfg.params.NF_VERSION).to(device)    

# remeshing utils
with open(remeshing_path, 'rb') as fp: 
    remeshing_dict = pkl.load(fp)
remeshing_relevant_faces = torch.tensor(remeshing_dict['smal_faces'][remeshing_dict['faceid_closest']], dtype=torch.long, device=device)
remeshing_relevant_barys = torch.tensor(remeshing_dict['barys_closest'], dtype=torch.float32, device=device)




# create path for output files
save_imgs_path = os.path.join(cfg.paths.ROOT_OUT_PATH, 'gradio_examples')
if not os.path.exists(save_imgs_path):
    os.makedirs(save_imgs_path)





def run_bite_inference(input_image, bbox=None, apply_ttopt=True,  dog_name="dog_model"):

    with open(loss_weight_path, 'r') as j: 
        losses = json.loads(j.read())
    shutil.copyfile(loss_weight_path, root_out_path_details + os.path.basename(loss_weight_path))

    # prepare dataset and dataset loader
    val_dataset, val_loader, len_val_dataset, test_name_list, stanext_data_info, stanext_acc_joints = get_single_crop_dataset_from_image(input_image, bbox=bbox)

    # summarize information for normalization 
    norm_dict = get_norm_dict(stanext_data_info, device)
    # get keypoint weights
    keypoint_weights = torch.tensor(stanext_data_info.keypoint_weights, dtype=torch.float)[None, :].to(device) 


    # prepare progress bar
    iterable = enumerate(val_loader) # the length of this iterator should be 1
    progress = None
    if False:        # not quiet:
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
        

    ind_img = 0
    name = (test_name_list[target_dict['index'][ind_img].long()]).replace('/', '__').split('.')[0]

    ind_img_tot += 1
    batch_size = 1

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
        # vert_colors = np.repeat(255*target_gc_class.detach().cpu().numpy()[0, :, None], 3, 1)
        # vert_colors[:, 2] = 255
        vert_colors = np.ones_like(np.repeat(target_gc_class.detach().cpu().numpy()[0, :, None], 3, 1)) * 255
        faces_prep = smal.faces.unsqueeze(0).expand((batch_size, -1, -1))
        # prepare target silhouette and keypoints, from stacked hourglass predictions
        target_hg_silh = res['hg_silh_prep'][ind_img, :, :].detach()
        target_kp_resh = res['hg_keyp_256'][ind_img, None, :, :].reshape((-1, 2)).detach()
        # find out if ground contact constraints should be used for the image at hand
        if res['isflat_prep'][ind_img] >= 0.5: # threshold should probably be set higher
            isflat = [True]
        else:
            isflat = [False] 
        if target_gc_class_remeshed_prep.sum() > 3:
            istouching = [True]
        else:
            istouching = [False]
        ignore_pose_optimization = False


    if not apply_ttopt:
        # get 3d smal model
        optimed_pose_with_glob = get_optimed_pose_with_glob(optimed_orient_6d, optimed_pose_6d)
        optimed_trans = torch.cat((optimed_trans_xy, optimed_trans_z), dim=1)
        smal_verts, keyp_3d, _ = smal(beta=optimed_betas, betas_limbs=optimed_betas_limbs, pose=optimed_pose_with_glob, vert_off_compact=optimed_vert_off_compact, trans=optimed_trans, keyp_conf='olive', get_skin=True)

        # save mesh
        my_mesh_tri = trimesh.Trimesh(vertices=smal_verts[0, ...].detach().cpu().numpy(), faces=faces_prep[0, ...].detach().cpu().numpy(), process=False,  maintain_order=True)
        my_mesh_tri.visual.vertex_colors = vert_colors
        # my_mesh_tri.export(root_out_path +  name + '_res_e000' + '.obj')

    else: 

        ##########################################################################################################
        # start optimizing for this image
        n_iter = 301    # how many iterations are desired? (+1)
        loop = range(n_iter)
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
            """
            if i==0:
                img_silh = Image.fromarray(np.uint8(255*pred_silh_images[0, 0, :, :].detach().cpu().numpy())).convert('RGB')
                img_silh.save(root_out_path_details +  name + '_silh_ainit.png')
                my_mesh_tri = trimesh.Trimesh(vertices=smal_verts[0, ...].detach().cpu().numpy(), faces=faces_prep[0, ...].detach().cpu().numpy(), process=False,  maintain_order=True)
                my_mesh_tri.export(root_out_path_details +  name + '_res_ainit.obj')
            """

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
            # loop.set_description(f"Body Fitting = {total_loss.item():.3f}")

            # save the result three times (0, 150, 300)
            if i == 300:  # if i % 150 == 0:    
                # save silhouette image
                img_silh = Image.fromarray(np.uint8(255*pred_silh_images[0, 0, :, :].detach().cpu().numpy())).convert('RGB')
                img_silh.save(root_out_path_details +  name + '_silh_e' + format(i, '03d') + '.png')
                # save image overlay
                visualizations = silh_renderer.get_visualization_nograd(smal_verts, faces_prep, optimed_camera_flength, color=0)
                pred_tex = visualizations[0, :, :, :].permute((1, 2, 0)).cpu().detach().numpy() / 256
                # out_path = root_out_path_details +  name + '_tex_pred_e' + format(i, '03d') + '.png'
                # plt.imsave(out_path, pred_tex)
                pred_tex_max = np.max(pred_tex, axis=2)
                out_path = root_out_path +  name + '_comp_pred_e' + format(i, '03d') + '.png'
                # save mesh
                my_mesh_tri = trimesh.Trimesh(vertices=smal_verts[0, ...].detach().cpu().numpy(), faces=faces_prep[0, ...].detach().cpu().numpy(), process=False,  maintain_order=True)
                my_mesh_tri.visual.vertex_colors = vert_colors
                # my_mesh_tri.export(root_out_path +  name + '_res_e' + format(i, '03d') + '.obj')
                # save focal length (together with the mesh this is enough to create an overlay in blender)
                # out_file_flength = root_out_path_details +  name + '_flength_e' + format(i, '03d') # + '.npz'
                # np.save(out_file_flength, optimed_camera_flength.detach().cpu().numpy())
            current_i += 1

    # prepare output mesh
    mesh = my_mesh_tri  # all_results[0]['mesh_posed']
    mesh.apply_transform([[-1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])
    result_path = os.path.join(save_imgs_path, dog_name)
    mesh.export(file_obj=result_path + '.glb')
    result_gltf = result_path + '.glb'
    return result_gltf





# -------------------------------------------------------------------------------------------------------------------- #

total_count = 0

def run_complete_inference(img_path_or_img, crop_choice, use_ttopt):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    global total_count
    total_count += 1
    print(dt_string + ' total count: ' + str(total_count))
    # depending on crop_choice: run faster r-cnn or take the input image directly
    if crop_choice == "input image is cropped":
        if isinstance(img_path_or_img, str):
            img = cv2.imread(img_path_or_img)
            output_interm_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            output_interm_image = img_path_or_img
        output_interm_bbox = None
    else:
        output_interm_image, output_interm_bbox = run_bbox_inference(img_path_or_img.copy())
    if use_ttopt == "enable test-time optimization":
        apply_ttopt = True
    else:
        apply_ttopt = False
    # run barc inference
    if img_path_or_img.dtype == str:
        dog_name = os.path.basename(img_path_or_img).split(".")[0]
    else:
        dog_name = "dog"

    result_gltf = run_bite_inference(img_path_or_img, output_interm_bbox, apply_ttopt, dog_name=dog_name)
    # add white border to image for nicer alignment
    output_interm_image_vis = np.concatenate((255*np.ones_like(output_interm_image), output_interm_image, 255*np.ones_like(output_interm_image)), axis=1)
    return [result_gltf, result_gltf, output_interm_image_vis]




########################################################################################################################

description = '''
# BITE

#### Project Page
* https://bite.is.tue.mpg.de/

#### Description
This is a demo for BITE (*B*eyond Priors for *I*mproved *T*hree-{D} Dog Pose *E*stimation). 
To run inference on one of the examples below, click on the desired image and push the submit button. Alternatively, you may upload one of your own images.
You can either submit a cropped image or choose the option to run a pretrained Faster R-CNN in order to obtain a bounding box. 
While we recommend enabeling test-time optimization (computation can take up to a minute), you have the possibility to skip it, which will lead to faster calculation (a few seconds) at the cost of less accurate results. 
<details>

<summary>More</summary>

#### Citation

```
@inproceedings{bite2023rueegg,
        title = {{BITE}: Beyond Priors for Improved Three-{D} Dog Pose Estimation},
        author = {R\"uegg, Nadine and Tripathi, Shashank and Schindler, Konrad and Black, Michael J. and Zuffi, Silvia},
        booktitle = {IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR)},
        pages = {8867-8876},
        year = {2023},
}
```

#### Image Sources
* Stanford extra image dataset
* Images from google search engine
    * https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRnx2sHnnLU3zy1XnJB7BvGUR9spmAh5bxTUg&usqp=CAU
    * https://www.westend61.de/en/imageView/CAVF56467/portrait-of-dog-lying-on-floor-at-home

#### Disclosure
The results shown in this demo are slightly improved compared to the ones depicted within our paper, as we apply a regularizer on the tail.

</details>

'''






example_images = sorted(glob.glob(os.path.join(os.path.dirname(__file__), '../', 'datasets', 'test_image_crops', '*.jpg')) + glob.glob(os.path.join(os.path.dirname(__file__), '../', 'datasets', 'test_image_crops', '*.jpeg')) + glob.glob(os.path.join(os.path.dirname(__file__), '../', 'datasets', 'test_image_crops', '*.png')))  
random.shuffle(example_images)
# example_images.reverse()
# examples = [[img, "input image is cropped"] for img in example_images]
examples = []
for img in example_images:
    if os.path.basename(img)[:2] == 'z_':
        examples.append([img, "use Faster R-CNN to get a bounding box", "enable test-time optimization"])
    else:
        examples.append([img, "input image is cropped", "enable test-time optimization"])

demo = gr.Interface(
    fn=run_complete_inference,
    description=description,
    inputs=[gr.Image(label="Input Image"),
        gr.Radio(["input image is cropped", "use Faster R-CNN to get a bounding box"], value="use Faster R-CNN to get a bounding box", label="Crop Choice"),
        gr.Radio(["enable test-time optimization", "skip test-time optimization"], value="enable test-time optimization", label="Test Time Optimization"),
    ],
    outputs=[
        gr.Model3D(
            clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model"),
        gr.File(label="Download 3D Model"),
        gr.Image(label="Bounding Box (Faster R-CNN prediction)"),

    ],
    examples=examples,
    thumbnail="bite_thumbnail.png",
    allow_flagging="never",
    cache_examples=False, #True,
    examples_per_page=14,
)

demo.launch(share=True)