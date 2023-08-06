
# evaluate test time optimization from refinement
# python src/test_time_optimization/evaluate_ttopt.py --workers 12 --save-images True --config refinement_cfg_test_withvertexwisegc_csaddnonflat.yaml --model-file-complete=cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0/checkpoint.pth.tar --ttopt-result-name ttoptv6_stanext_v16b 

# python src/test_time_optimization/evaluate_ttopt.py --workers 12 --save-images True --config refinement_cfg_test_withvertexwisegc_csaddnonflat.yaml --model-file-complete=cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0/checkpoint.pth.tar --ttopt-result-name ttoptv6_stanext_v16 



import argparse
import os.path
import json
import numpy as np
import pickle as pkl
from distutils.util import strtobool
import torch
from torch import nn
import torch.backends.cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import pytorch3d as p3d
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from combined_model.train_main_image_to_3d_wbr_withref import do_validation_epoch
# from combined_model.model_shape_v7 import ModelImageTo3d_withshape_withproj  
# from combined_model.model_shape_v7_withref import ModelImageTo3d_withshape_withproj 
from combined_model.model_shape_v7_withref_withgraphcnn import ModelImageTo3d_withshape_withproj 

from combined_model.loss_image_to_3d_withbreedrel import Loss
from combined_model.loss_image_to_3d_refinement import LossRef
from configs.barc_cfg_defaults import get_cfg_defaults, update_cfg_global_with_yaml, get_cfg_global_updated

from lifting_to_3d.utils.geometry_utils import rot6d_to_rotmat, rotmat_to_rot6d  # , batch_rot2aa, geodesic_loss_R


# from test_time_optimization.utils_ttopt import get_evaluation_dataset, get_norm_dict
from stacked_hourglass.datasets.utils_dataset_selection import get_evaluation_dataset, get_norm_dict

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

from metrics.metrics import Metrics
from configs.SMAL_configs import EVAL_KEYPOINTS, KEYPOINT_GROUPS


ROOT_LOSS_WEIGH_PATH = '/is/cluster/work/nrueegg/icon_pifu_related/barc_for_bite/src/configs/ttopt_loss_weights/'



def main(args):

    # load configs
    #   step 1: load default configs
    #   step 2: load updates from .yaml file
    path_config = os.path.join(get_cfg_defaults().barc_dir, 'src', 'configs', args.config)
    update_cfg_global_with_yaml(path_config)
    cfg = get_cfg_global_updated()

    pck_thresh = 0.15
    print('pck_thresh: ' + str(pck_thresh))




    ROOT_IN_PATH = '/is/cluster/work/nrueegg/icon_pifu_related/barc_for_bite/results/results_ttopt/' + args.ttopt_result_name + '/'      # ttoptv6_debug_x8/'    
    ROOT_IN_PATH_DETAIL = ROOT_IN_PATH + 'details/'

    ROOT_OUT_PATH = ROOT_IN_PATH + 'evaluation/'
    if not os.path.exists(ROOT_OUT_PATH): os.makedirs(ROOT_OUT_PATH)









    
    # NEW!!!
    logscale_part_list = ['legs_l', 'legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l'] 
    # logscale_part_list = ['front_legs_l', 'front_legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l', 'back_legs_l', 'back_legs_f'] 


    # Select the hardware device to use for training.
    if torch.cuda.is_available() and cfg.device=='cuda':
        device = torch.device('cuda', torch.cuda.current_device())
        torch.backends.cudnn.benchmark = False      # True
    else:
        device = torch.device('cpu')

    print('structure_pose_net: ' + cfg.params.STRUCTURE_POSE_NET)
    print('refinement network type: ' + cfg.params.REF_NET_TYPE)
    print('smal_model_type: ' + cfg.smal.SMAL_MODEL_TYPE)

    path_model_file_complete = os.path.join(cfg.paths.ROOT_CHECKPOINT_PATH, args.model_file_complete) 

    # Disable gradient calculations.
    # torch.set_grad_enabled(False)


    # prepare dataset and dataset loadr
    val_dataset, val_loader, len_val_dataset, test_name_list, stanext_data_info, stanext_acc_joints = get_evaluation_dataset(cfg.data.DATASET, cfg.data.VAL_OPT, cfg.data.V12, cfg.optim.BATCH_SIZE, args.workers)
    len_data = len_val_dataset
    # summarize information for normalization 
    norm_dict = get_norm_dict(stanext_data_info, device)

    # prepare complete model
    bite_model = BITEInferenceModel(cfg, path_model_file_complete, norm_dict)
    # smal_model_type = bite_model.complete_model.smal.smal_model_type
    smal_model_type = bite_model.smal_model_type
    smal = SMAL(smal_model_type=smal_model_type, template_name='neutral', logscale_part_list=logscale_part_list).to(device)    
    silh_renderer = SilhRenderer(image_size=256).to(device)    



    # ----------------------------------------------------------------------------------

    summary = {}
    summary['pck'] = np.zeros((len_data))
    summary['pck_by_part'] = {group:np.zeros((len_data)) for group in KEYPOINT_GROUPS}
    summary['acc_sil_2d'] = np.zeros(len_data)











    # Put the model in training mode.
    # model.train()
    # prepare progress bar
    iterable = enumerate(val_loader)
    progress = None
    if True:        # not quiet:
        progress = tqdm(iterable, desc='Train', total=len(val_loader), ascii=True, leave=False)
        iterable = progress
    ind_img_tot = 0
    # prepare variables, put them on the right device

    my_step = 0
    batch_size = cfg.optim.BATCH_SIZE

    for index, (input, target_dict) in iterable:
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
        #   -> here from barc, but could also be saved and loaded
        preds_dict = bite_model.get_all_results(input)
        res_normal_and_ref = bite_model.get_selected_results(preds_dict=preds_dict, result_networks=['normal', 'ref'])
        res = bite_model.get_selected_results(preds_dict=preds_dict, result_networks=['ref'])['ref']

        # --------------------------------------------------------------------

        # ind_img = 0

        batch_verts_smal = []
        batch_faces_prep = []
        batch_optimed_camera_flength = []



        for ind_img in range(input.shape[0]):
            name = (test_name_list[target_dict['index'][ind_img].long()]).replace('/', '__').split('.')[0]

            print('ind_img_tot: ' + str(ind_img_tot) + '   -> ' + name)
            ind_img_tot += 1

            e_name = 'e000'     # 'e300'

            npy_file = ROOT_IN_PATH_DETAIL + name + '_flength_' + e_name +'.npy'
            flength = np.load(npy_file)
            optimed_camera_flength = torch.tensor(flength, device=device)

            obj_file = ROOT_IN_PATH + name + '_res_' + e_name +'.obj'

            verts, faces, aux = p3d.io.load_obj(obj_file)
            verts_smal = verts[None, ...].to(device)
            faces_prep = faces.verts_idx[None, ...].to(device)
            batch_verts_smal.append(verts_smal)
            batch_faces_prep.append(faces_prep)
            batch_optimed_camera_flength.append(optimed_camera_flength)


        # import pdb; pdb.set_trace()

        verts_smal = torch.cat(batch_verts_smal, dim=0)
        faces_prep = torch.cat(batch_faces_prep, dim=0)
        optimed_camera_flength = torch.cat(batch_optimed_camera_flength, dim=0)

        # get keypoint locations from mesh vertices
        keyp_3d = smal.get_joints_from_verts(verts_smal, keyp_conf='olive')
        

        # render silhouette and keypoints
        pred_silh_images, pred_keyp_raw = silh_renderer(vertices=verts_smal, points=keyp_3d, faces=faces_prep, focal_lengths=optimed_camera_flength)
        pred_keyp = pred_keyp_raw[:, :24, :]



        # --------------- calculate iou and pck values --------------------

        gt_keypoints_256 = target_dict['tpts'][:, :, :2] / 64. * (256. - 1)
        gt_keypoints = torch.cat((gt_keypoints_256, target_dict['tpts'][:, :, 2:3]), dim=2)    
        # prepare silhouette for IoU calculation - predicted as well as ground truth
        has_seg = target_dict['has_seg']
        img_border_mask = target_dict['img_border_mask'][:, 0, :, :]
        gtseg = target_dict['silh']
        synth_silhouettes = pred_silh_images[:, 0, :, :]   # pred_silh[:, 0, :, :]       # output_reproj['silh']
        synth_silhouettes[synth_silhouettes>0.5] = 1
        synth_silhouettes[synth_silhouettes<0.5] = 0
        # calculate PCK as well as IoU (similar to WLDO)
        preds = {}
        preds['acc_PCK'] = Metrics.PCK(
            pred_keyp, gt_keypoints, 
            gtseg, has_seg, idxs=EVAL_KEYPOINTS,
            thresh_range=[pck_thresh],       # [0.15],
        )
        preds['acc_IOU'] = Metrics.IOU(
            synth_silhouettes, gtseg, 
            img_border_mask, mask=has_seg
        )
        for group, group_kps in KEYPOINT_GROUPS.items():
            preds[f'{group}_PCK'] = Metrics.PCK(
                pred_keyp, gt_keypoints, gtseg, has_seg, 
                thresh_range=[pck_thresh],       # [0.15],
                idxs=group_kps
            )

        curr_batch_size = pred_keyp.shape[0]
        if not (preds['acc_PCK'].data.cpu().numpy().shape == (summary['pck'][my_step * batch_size:my_step * batch_size + curr_batch_size]).shape):
            import pdb; pdb.set_trace()
        summary['pck'][my_step * batch_size:my_step * batch_size + curr_batch_size] = preds['acc_PCK'].data.cpu().numpy()
        summary['acc_sil_2d'][my_step * batch_size:my_step * batch_size + curr_batch_size] = preds['acc_IOU'].data.cpu().numpy()
        for part in summary['pck_by_part']:
            summary['pck_by_part'][part][my_step * batch_size:my_step * batch_size + curr_batch_size] = preds[f'{part}_PCK'].data.cpu().numpy()




        my_step += 1





    # import pdb; pdb.set_trace()






    iou = np.nanmean(summary['acc_sil_2d'])
    pck = np.nanmean(summary['pck'])
    pck_legs = np.nanmean(summary['pck_by_part']['legs'])
    pck_tail = np.nanmean(summary['pck_by_part']['tail'])
    pck_ears = np.nanmean(summary['pck_by_part']['ears'])
    pck_face = np.nanmean(summary['pck_by_part']['face'])
    print('------------------------------------------------')
    print("iou:         {:.2f}".format(iou*100))
    print('                                                ')
    print("pck:         {:.2f}".format(pck*100))
    print('                                                ')
    print("pck_legs:    {:.2f}".format(pck_legs*100))
    print("pck_tail:    {:.2f}".format(pck_tail*100))
    print("pck_ears:    {:.2f}".format(pck_ears*100))
    print("pck_face:    {:.2f}".format(pck_face*100))
    print('------------------------------------------------')
    # save results in a .txt file
    with open(ROOT_OUT_PATH + "a_evaluation_" + e_name + ".txt", "w") as text_file:
        print("iou:         {:.2f}".format(iou*100), file=text_file)
        print("pck:         {:.2f}".format(pck*100), file=text_file)
        print("pck_legs:    {:.2f}".format(pck_legs*100), file=text_file)
        print("pck_tail:    {:.2f}".format(pck_tail*100), file=text_file)
        print("pck_ears:    {:.2f}".format(pck_ears*100), file=text_file)
        print("pck_face:    {:.2f}".format(pck_face*100), file=text_file)




































if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a stacked hourglass model.')
    parser.add_argument('--model-file-complete', default='', type=str, metavar='PATH',
                        help='path to saved model weights')
    parser.add_argument('--ttopt-result-name', default='', type=str, metavar='PATH',
                        help='path to saved ttopt results')
    parser.add_argument('-cg', '--config', default='barc_cfg_test.yaml', type=str, metavar='PATH',
                        help='name of config file (default: barc_cfg_test.yaml within src/configs folder)')
    parser.add_argument('--save-images', default='True', type=lambda x: bool(strtobool(x)),
                        help='bool indicating if images should be saved')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--metrics', '-m', metavar='METRICS', default='all',
                        choices=['all', None],
                        help='model architecture')
    main(parser.parse_args())
