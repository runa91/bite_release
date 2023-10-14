
import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.nn.parallel
from tqdm import tqdm
import os
import pathlib
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch
import trimesh

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from stacked_hourglass.utils.evaluation import accuracy, AverageMeter, final_preds, get_preds, get_preds_soft
from stacked_hourglass.utils.visualization import save_input_image_with_keypoints, save_input_image
from metrics.metrics import Metrics
from configs.SMAL_configs import EVAL_KEYPOINTS, KEYPOINT_GROUPS


# GOAL: have all the functions from the validation and visual epoch together

def eval_save_visualizations_and_meshes(model, input, data_info, target_dict, test_name_list, vertices_smal, hg_keyp_norm, hg_keyp_scores, zz, betas, betas_limbs, pose_rotmat, trans, flength, pred_keyp, pred_silh, save_imgs_path, prefix, index, render_all=False):
    device = input.device
    curr_batch_size = input.shape[0]
    # render predicted 3d models
    visualizations = model.render_vis_nograd(vertices=vertices_smal,
                                            focal_lengths=flength,
                                            color=0)        # color=2)
    for ind_img in range(len(target_dict['index'])):    
        try: 
            # import pdb; pdb.set_trace()
            if test_name_list is not None:
                img_name = test_name_list[int(target_dict['index'][ind_img].cpu().detach().numpy())].replace('/', '_')
                img_name = img_name.split('.')[0]
            else:
                img_name = str(index) + '_' + str(ind_img)
            # save image with predicted keypoints
            out_path = save_imgs_path + '/keypoints_pred_' + img_name + '.png'
            pred_unp = (hg_keyp_norm[ind_img, :, :] + 1.) / 2 * (data_info.image_size - 1)
            pred_unp_maxval = hg_keyp_scores[ind_img, :, :]
            pred_unp_prep = torch.cat((pred_unp, pred_unp_maxval), 1)
            inp_img = input[ind_img, :, :, :].detach().clone()
            save_input_image_with_keypoints(inp_img, pred_unp_prep, out_path=out_path, threshold=0.1, print_scores=True, ratio_in_out=1.0)    # threshold=0.3
            # save predicted 3d model (front view)
            pred_tex = visualizations[ind_img, :, :, :].permute((1, 2, 0)).cpu().detach().numpy() / 256
            pred_tex_max = np.max(pred_tex, axis=2)
            out_path = save_imgs_path + '/' + prefix + 'tex_pred_' + img_name + '.png'
            plt.imsave(out_path, pred_tex)
            input_image = input[ind_img, :, :, :].detach().clone()
            for t, m, s in zip(input_image, data_info.rgb_mean, data_info.rgb_stddev): t.add_(m)
            input_image_np = input_image.detach().cpu().numpy().transpose(1, 2, 0) 
            im_masked = cv2.addWeighted(input_image_np,0.2,pred_tex,0.8,0)
            im_masked[pred_tex_max<0.01, :] = input_image_np[pred_tex_max<0.01, :]
            out_path = save_imgs_path + '/' + prefix + 'comp_pred_' + img_name + '.png'
            plt.imsave(out_path, im_masked)
            # save predicted 3d model (side view)
            vertices_cent = vertices_smal - vertices_smal.mean(dim=1)[:, None, :]
            roll = np.pi / 2 * torch.ones(1).float().to(device)
            pitch = np.pi / 2 * torch.ones(1).float().to(device)
            tensor_0 = torch.zeros(1).float().to(device)
            tensor_1 = torch.ones(1).float().to(device)
            RX = torch.stack([torch.stack([tensor_1, tensor_0, tensor_0]), torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3,3)
            RY = torch.stack([
                torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                torch.stack([tensor_0, tensor_1, tensor_0]),
                torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3,3)
            vertices_rot = (torch.matmul(RY, vertices_cent.reshape((-1, 3))[:, :, None])).reshape((curr_batch_size, -1, 3))
            vertices_rot[:, :, 2] = vertices_rot[:, :, 2] + torch.ones_like(vertices_rot[:, :, 2]) * 20     # 18     # *16

            visualizations_rot = model.render_vis_nograd(vertices=vertices_rot,
                                                    focal_lengths=flength,
                                                    color=0)        # 2)
            pred_tex = visualizations_rot[ind_img, :, :, :].permute((1, 2, 0)).cpu().detach().numpy() / 256
            pred_tex_max = np.max(pred_tex, axis=2)
            out_path = save_imgs_path + '/' + prefix + 'rot_tex_pred_' + img_name + '.png'
            plt.imsave(out_path, pred_tex)
            if render_all:
                # save input image
                inp_img = input[ind_img, :, :, :].detach().clone()
                out_path = save_imgs_path + '/image_' + img_name + '.png'
                save_input_image(inp_img, out_path)
                # save mesh
                V_posed = vertices_smal[ind_img, :, :].detach().cpu().numpy()
                Faces = model.smal.f
                mesh_posed = trimesh.Trimesh(vertices=V_posed, faces=Faces, process=False,  maintain_order=True)
                mesh_posed.export(save_imgs_path + '/' + prefix + 'mesh_posed_' + img_name + '.obj')
        except: 
            print('dont save an image')


def eval_prepare_pck_and_iou(model, input, data_info, target_dict, test_name_list, vertices_smal, hg_keyp_norm, hg_keyp_scores, zz, betas, betas_limbs, pose_rotmat, trans, flength, pred_keyp, pred_silh, save_imgs_path, prefix, index, pck_thresh, progress=None, skip_pck_and_iou=False):
    preds = {}
    preds['betas'] = betas.cpu().detach().numpy()
    preds['betas_limbs'] = betas_limbs.cpu().detach().numpy()
    preds['z'] = zz.cpu().detach().numpy()
    preds['pose_rotmat'] = pose_rotmat.cpu().detach().numpy()
    preds['flength'] = flength.cpu().detach().numpy()
    preds['trans'] = trans.cpu().detach().numpy()
    preds['breed_index'] = target_dict['breed_index'].cpu().detach().numpy().reshape((-1))
    img_names = []
    for ind_img2 in range(0, betas.shape[0]):
        if test_name_list is not None:
            img_name2 = test_name_list[int(target_dict['index'][ind_img2].cpu().detach().numpy())].replace('/', '_')
            img_name2 = img_name2.split('.')[0]
        else:
            img_name2 = str(index) + '_' + str(ind_img2)
        img_names.append(img_name2)
    preds['image_names'] = img_names
    if not skip_pck_and_iou:
        # prepare keypoints for PCK calculation - predicted as well as ground truth
        # pred_keyp = output_reproj['keyp_2d']   # 256
        gt_keypoints_256 = target_dict['tpts'][:, :, :2] / 64. * (256. - 1)
        # gt_keypoints_norm = gt_keypoints_256 / 256 / 0.5 - 1
        gt_keypoints = torch.cat((gt_keypoints_256, target_dict['tpts'][:, :, 2:3]), dim=2)     # gt_keypoints_norm
        # prepare silhouette for IoU calculation - predicted as well as ground truth
        has_seg = target_dict['has_seg']
        img_border_mask = target_dict['img_border_mask'][:, 0, :, :]
        gtseg = target_dict['silh']
        synth_silhouettes = pred_silh[:, 0, :, :]       # output_reproj['silh']
        synth_silhouettes[synth_silhouettes>0.5] = 1
        synth_silhouettes[synth_silhouettes<0.5] = 0
        # calculate PCK as well as IoU (similar to WLDO)
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
    return preds


# preds['acc_PCK'] = Metrics.PCK(pred_keyp, gt_keypoints, gtseg, has_seg, idxs=EVAL_KEYPOINTS, thresh_range=[pck_thresh])
# preds['acc_IOU'] = Metrics.IOU(synth_silhouettes, gtseg, img_border_mask, mask=has_seg)
#############################

def eval_add_preds_to_summary(summary, preds, my_step, batch_size, curr_batch_size, skip_pck_and_iou=False):
    if not skip_pck_and_iou:
        if not (preds['acc_PCK'].data.cpu().numpy().shape == (summary['pck'][my_step * batch_size:my_step * batch_size + curr_batch_size]).shape):
            import pdb; pdb.set_trace()
        summary['pck'][my_step * batch_size:my_step * batch_size + curr_batch_size] = preds['acc_PCK'].data.cpu().numpy()
        summary['acc_sil_2d'][my_step * batch_size:my_step * batch_size + curr_batch_size] = preds['acc_IOU'].data.cpu().numpy()
        for part in summary['pck_by_part']:
            summary['pck_by_part'][part][my_step * batch_size:my_step * batch_size + curr_batch_size] = preds[f'{part}_PCK'].data.cpu().numpy()
    summary['betas'][my_step * batch_size:my_step * batch_size + curr_batch_size, ...] = preds['betas'] 
    summary['betas_limbs'][my_step * batch_size:my_step * batch_size + curr_batch_size, ...] = preds['betas_limbs'] 
    summary['z'][my_step * batch_size:my_step * batch_size + curr_batch_size, ...] = preds['z'] 
    summary['pose_rotmat'][my_step * batch_size:my_step * batch_size + curr_batch_size, ...] = preds['pose_rotmat'] 
    summary['flength'][my_step * batch_size:my_step * batch_size + curr_batch_size, ...] = preds['flength'] 
    summary['trans'][my_step * batch_size:my_step * batch_size + curr_batch_size, ...] = preds['trans'] 
    summary['breed_indices'][my_step * batch_size:my_step * batch_size + curr_batch_size] = preds['breed_index']
    summary['image_names'].extend(preds['image_names'])
    return


def get_triangle_faces_from_pyvista_poly(poly):
    """Fetch all triangle faces."""
    stream = poly.faces
    tris = []
    i = 0
    while i < len(stream):
        n = stream[i]
        if n != 3:
            i += n + 1
            continue
        stop = i + n + 1
        tris.append(stream[i+1:stop])
        i = stop
    return np.array(tris)