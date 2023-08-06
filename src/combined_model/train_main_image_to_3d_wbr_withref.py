
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
import pickle as pkl
import csv
from scipy.spatial.transform import Rotation as R_sc


import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from stacked_hourglass.utils.evaluation import accuracy, AverageMeter, final_preds, get_preds, get_preds_soft
from stacked_hourglass.utils.visualization import save_input_image_with_keypoints, save_input_image
from metrics.metrics import Metrics
from configs.SMAL_configs import EVAL_KEYPOINTS, KEYPOINT_GROUPS, SMAL_KEYPOINT_NAMES_FOR_3D_EVAL, SMAL_KEYPOINT_INDICES_FOR_3D_EVAL, SMAL_KEYPOINT_WHICHTOUSE_FOR_3D_EVAL
from combined_model.helper import eval_save_visualizations_and_meshes, eval_prepare_pck_and_iou, eval_add_preds_to_summary

from smal_pytorch.smal_model.smal_torch_new import SMAL     # for gc visualization
from src.combined_model.loss_utils.loss_utils import fit_plane
# from src.evaluation.sketchfab_evaluation.alignment_utils.calculate_v2v_error_release import compute_similarity_transform
# from src.evaluation.sketchfab_evaluation.alignment_utils.calculate_alignment_error import calculate_alignemnt_errors

# ---------------------------------------------------------------------------------------------------------------------------
def do_training_epoch(train_loader, model, loss_module, loss_module_ref, device, data_info, optimiser, quiet=False, acc_joints=None, weight_dict=None, weight_dict_ref=None):
    losses = AverageMeter()
    losses_keyp = AverageMeter()
    losses_silh = AverageMeter()
    losses_shape = AverageMeter()
    losses_pose = AverageMeter()
    losses_class = AverageMeter()
    losses_breed = AverageMeter()
    losses_partseg = AverageMeter()
    losses_ref_keyp = AverageMeter()
    losses_ref_silh = AverageMeter()
    losses_ref_pose = AverageMeter()
    losses_ref_reg = AverageMeter()
    accuracies = AverageMeter()
    # Put the model in training mode.
    model.train()
    # prepare progress bar
    iterable = enumerate(train_loader)
    progress = None
    if not quiet:
        progress = tqdm(iterable, desc='Train', total=len(train_loader), ascii=True, leave=False)
        iterable = progress
    # information for normalization 
    norm_dict = {
        'pose_rot6d_mean': torch.from_numpy(data_info.pose_rot6d_mean).float().to(device),
        'trans_mean': torch.from_numpy(data_info.trans_mean).float().to(device),
        'trans_std': torch.from_numpy(data_info.trans_std).float().to(device),
        'flength_mean': torch.from_numpy(data_info.flength_mean).float().to(device),
        'flength_std': torch.from_numpy(data_info.flength_std).float().to(device)}
    # prepare variables, put them on the right device
    for i, (input, target_dict) in iterable:
        batch_size = input.shape[0]
        for key in target_dict.keys(): 
            if key == 'breed_index':
                target_dict[key] = target_dict[key].long().to(device)
            elif key in ['index', 'pts', 'tpts', 'target_weight', 'silh', 'silh_distmat_tofg', 'silh_distmat_tobg', 'sim_breed_index', 'img_border_mask']:
                target_dict[key] = target_dict[key].float().to(device)
            elif key in ['has_seg', 'gc']:
                target_dict[key] = target_dict[key].to(device)
            else:
                pass
        input = input.float().to(device)

        # ----------------------- do training step -----------------------
        assert model.training, 'model must be in training mode.'
        with torch.enable_grad():
            # ----- forward pass -----  
            output, output_unnorm, output_reproj, output_ref, output_ref_comp = model(input, norm_dict=norm_dict)        
            # ----- loss -----
            # --- from main network
            loss, loss_dict = loss_module(output_reproj=output_reproj, 
                target_dict=target_dict, 
                weight_dict=weight_dict)
            # ---from refinement network
            loss_ref, loss_dict_ref = loss_module_ref(output_ref=output_ref, 
                output_ref_comp=output_ref_comp,
                target_dict=target_dict, 
                weight_dict_ref=weight_dict_ref)
            loss_total = loss + loss_ref
            # ----- backward pass and parameter update -----
            optimiser.zero_grad()
            loss_total.backward()
            optimiser.step()
        # ----------------------------------------------------------------

        # prepare losses for progress bar
        bs_fake = 1     # batch_size
        losses.update(loss_dict['loss'] + loss_dict_ref['loss'], bs_fake)
        losses_keyp.update(loss_dict['loss_keyp_weighted'], bs_fake)
        losses_silh.update(loss_dict['loss_silh_weighted'], bs_fake)
        losses_shape.update(loss_dict['loss_shape_weighted'], bs_fake)
        losses_pose.update(loss_dict['loss_poseprior_weighted'], bs_fake)   
        losses_class.update(loss_dict['loss_class_weighted'], bs_fake)
        losses_breed.update(loss_dict['loss_breed_weighted'], bs_fake)
        losses_partseg.update(loss_dict['loss_partseg_weighted'], bs_fake)
        losses_ref_keyp.update(loss_dict_ref['keyp_ref'], bs_fake)
        losses_ref_silh.update(loss_dict_ref['silh_ref'], bs_fake)
        loss_ref_pose = 0
        for l_name in ['pose_legs_side', 'pose_legs_tors', 'pose_tail_side', 'pose_tail_tors', 'pose_spine_side', 'pose_spine_tors']:
            if l_name in loss_dict_ref.keys():
                loss_ref_pose += loss_dict_ref[l_name]
        losses_ref_pose.update(loss_ref_pose, bs_fake)
        loss_ref_reg = 0
        for l_name in ['reg_trans', 'reg_flength', 'reg_pose']:
            if l_name in loss_dict_ref.keys():
                loss_ref_reg += loss_dict_ref[l_name]
        losses_ref_reg.update(loss_ref_reg, bs_fake)
        acc = - loss_dict['loss_keyp_weighted']     # this will be used to keep track of the 'best model'
        accuracies.update(acc, bs_fake)
        # Show losses as part of the progress bar.
        if progress is not None:
            my_string = 'Loss: {loss:0.4f}, loss_keyp: {loss_keyp:0.4f}, loss_silh: {loss_silh:0.4f}, loss_partseg: {loss_partseg:0.4f}, loss_shape: {loss_shape:0.4f}, loss_pose: {loss_pose:0.4f}, loss_class: {loss_class:0.4f}, loss_breed: {loss_breed:0.4f}, loss_ref_keyp: {loss_ref_keyp:0.4f}, loss_ref_silh: {loss_ref_silh:0.4f}, loss_ref_pose: {loss_ref_pose:0.4f}, loss_ref_reg: {loss_ref_reg:0.4f}'.format(
                loss=losses.avg,
                loss_keyp=losses_keyp.avg,
                loss_silh=losses_silh.avg,
                loss_shape=losses_shape.avg,
                loss_pose=losses_pose.avg,
                loss_class=losses_class.avg,
                loss_breed=losses_breed.avg,
                loss_partseg=losses_partseg.avg,
                loss_ref_keyp=losses_ref_keyp.avg,
                loss_ref_silh=losses_ref_silh.avg,
                loss_ref_pose=losses_ref_pose.avg,
                loss_ref_reg=losses_ref_reg.avg)
            my_string_short = 'Loss: {loss:0.4f}, loss_keyp: {loss_keyp:0.4f}, loss_silh: {loss_silh:0.4f}, loss_ref_keyp: {loss_ref_keyp:0.4f}, loss_ref_silh: {loss_ref_silh:0.4f}, loss_ref_pose: {loss_ref_pose:0.4f}, loss_ref_reg: {loss_ref_reg:0.4f}'.format(
                loss=losses.avg,
                loss_keyp=losses_keyp.avg,
                loss_silh=losses_silh.avg,
                loss_ref_keyp=losses_ref_keyp.avg,
                loss_ref_silh=losses_ref_silh.avg,
                loss_ref_pose=losses_ref_pose.avg,
                loss_ref_reg=losses_ref_reg.avg)
            progress.set_postfix_str(my_string_short)

    return my_string, accuracies.avg       


# ---------------------------------------------------------------------------------------------------------------------------
def do_validation_epoch(val_loader, model, loss_module, loss_module_ref, device, data_info, flip=False, quiet=False, acc_joints=None, save_imgs_path=None, weight_dict=None, weight_dict_ref=None, metrics=None, val_opt='default', test_name_list=None, render_all=False, pck_thresh=0.15, len_dataset=None):
    losses = AverageMeter()
    losses_keyp = AverageMeter()
    losses_silh = AverageMeter()
    losses_shape = AverageMeter()
    losses_pose = AverageMeter()
    losses_class = AverageMeter()
    losses_breed = AverageMeter()
    losses_partseg = AverageMeter()
    losses_ref_keyp = AverageMeter()
    losses_ref_silh = AverageMeter()
    losses_ref_pose = AverageMeter()
    losses_ref_reg = AverageMeter()
    accuracies = AverageMeter()
    if save_imgs_path is not None:
        pathlib.Path(save_imgs_path).mkdir(parents=True, exist_ok=True) 
    # Put the model in evaluation mode.
    model.eval()
    # prepare progress bar
    iterable = enumerate(val_loader)
    progress = None
    if not quiet:
        progress = tqdm(iterable, desc='Valid', total=len(val_loader), ascii=True, leave=False)
        iterable = progress
    # summarize information for normalization 
    norm_dict = {
        'pose_rot6d_mean': torch.from_numpy(data_info.pose_rot6d_mean).float().to(device),
        'trans_mean': torch.from_numpy(data_info.trans_mean).float().to(device),
        'trans_std': torch.from_numpy(data_info.trans_std).float().to(device),
        'flength_mean': torch.from_numpy(data_info.flength_mean).float().to(device),
        'flength_std': torch.from_numpy(data_info.flength_std).float().to(device)}
    batch_size = val_loader.batch_size

    return_mesh_with_gt_groundplane = True
    if return_mesh_with_gt_groundplane:
        remeshing_path = '/is/cluster/work/nrueegg/icon_pifu_related/barc_for_bite/data/smal_data_remeshed/uniform_surface_sampling/my_smpl_39dogsnorm_Jr_4_dog_remesh4000_info.pkl'
        with open(remeshing_path, 'rb') as fp: 
            remeshing_dict = pkl.load(fp)
        remeshing_relevant_faces = torch.tensor(remeshing_dict['smal_faces'][remeshing_dict['faceid_closest']], dtype=torch.long, device=device)
        remeshing_relevant_barys = torch.tensor(remeshing_dict['barys_closest'], dtype=torch.float32, device=device)


    # from smal_pytorch.smal_model.smal_torch_new import SMAL
    print('start: load smal default model (barc), but only for vertices')
    smal = SMAL()
    print('end: load smal default model (barc), but only for vertices')
    smal_template_verts = smal.v_template.detach().cpu().numpy()
    smal_faces = smal.faces.detach().cpu().numpy()

    
    my_step = 0
    for index, (input, target_dict) in iterable:

        # prepare variables, put them on the right device
        curr_batch_size = input.shape[0]
        for key in target_dict.keys(): 
            if key == 'breed_index':
                target_dict[key] = target_dict[key].long().to(device)
            elif key in ['index', 'pts', 'tpts', 'target_weight', 'silh', 'silh_distmat_tofg', 'silh_distmat_tobg', 'sim_breed_index', 'img_border_mask']:
                target_dict[key] = target_dict[key].float().to(device)
            elif key in ['has_seg', 'gc']:
                target_dict[key] = target_dict[key].to(device)
            else:
                pass
        input = input.float().to(device)

        # ----------------------- do validation step -----------------------
        with torch.no_grad():
            # ----- forward pass -----  
            # output: (['pose', 'flength', 'trans', 'keypoints_norm', 'keypoints_scores'])
            # output_unnorm: (['pose_rotmat', 'flength', 'trans', 'keypoints'])
            # output_reproj: (['vertices_smal', 'torch_meshes', 'keyp_3d', 'keyp_2d', 'silh', 'betas', 'pose_rot6d', 'dog_breed', 'shapedirs', 'z', 'flength_unnorm', 'flength'])
            # target_dict: (['index', 'center', 'scale', 'pts', 'tpts', 'target_weight', 'breed_index', 'sim_breed_index', 'ind_dataset', 'silh'])
            output, output_unnorm, output_reproj, output_ref, output_ref_comp = model(input, norm_dict=norm_dict)        
            # ----- loss -----
            if metrics == 'no_loss':
                # --- from main network
                loss, loss_dict = loss_module(output_reproj=output_reproj, 
                    target_dict=target_dict, 
                    weight_dict=weight_dict)
                # ---from refinement network
                loss_ref, loss_dict_ref = loss_module_ref(output_ref=output_ref, 
                    output_ref_comp=output_ref_comp,
                    target_dict=target_dict, 
                    weight_dict_ref=weight_dict_ref)
                loss_total = loss + loss_ref

        # ----------------------------------------------------------------


        for result_network in ['normal', 'ref']:
            # variabled that are not refined
            hg_keyp_norm = output['keypoints_norm']
            hg_keyp_scores = output['keypoints_scores']
            betas = output_reproj['betas']
            betas_limbs = output_reproj['betas_limbs']
            zz = output_reproj['z']
            if result_network == 'normal':
                # STEP 1: normal network
                vertices_smal = output_reproj['vertices_smal']
                flength = output_unnorm['flength']
                pose_rotmat = output_unnorm['pose_rotmat']
                trans = output_unnorm['trans']
                pred_keyp = output_reproj['keyp_2d']
                pred_silh = output_reproj['silh']
                prefix = 'normal_'
            else:
                # STEP 1: refinement network
                vertices_smal = output_ref['vertices_smal']
                flength = output_ref['flength']
                pose_rotmat = output_ref['pose_rotmat']
                trans = output_ref['trans']
                pred_keyp = output_ref['keyp_2d']
                pred_silh = output_ref['silh']
                prefix = 'ref_'
                if return_mesh_with_gt_groundplane and 'gc' in target_dict.keys():
                    bs = vertices_smal.shape[0]
                    target_gc_class = target_dict['gc'][:, :, 0]
                    sel_verts = torch.index_select(output_ref['vertices_smal'], dim=1, index=remeshing_relevant_faces.reshape((-1))).reshape((bs, remeshing_relevant_faces.shape[0], 3, 3))
                    verts_remeshed = torch.einsum('ij,aijk->aik', remeshing_relevant_barys, sel_verts)
                    target_gc_class_remeshed = torch.einsum('ij,aij->ai', remeshing_relevant_barys, target_gc_class[:, remeshing_relevant_faces].to(device=device, dtype=torch.float32))
                    target_gc_class_remeshed_prep = torch.round(target_gc_class_remeshed).to(torch.long)





                # import pdb; pdb.set_trace()

                # new for vertex wise ground contact
                if (not model.graphcnn_type == 'inexistent') and (save_imgs_path is not None):
                    # import pdb; pdb.set_trace()

                    sm = torch.nn.Softmax(dim=2)
                    ground_contact_probs = sm(output_ref['vertexwise_ground_contact'])

                    for ind_img in range(ground_contact_probs.shape[0]):
                        # ind_img = 0
                        if test_name_list is not None:
                            img_name = test_name_list[int(target_dict['index'][ind_img].cpu().detach().numpy())].replace('/', '_')
                            img_name = img_name.split('.')[0]
                        else:
                            img_name = str(index) + '_' + str(ind_img)
                        out_path_gcmesh = save_imgs_path + '/' + prefix + 'gcmesh_' + img_name + '.obj'

                        gc_prob = ground_contact_probs[ind_img, :, 1]   # contact probability
                        vert_colors = np.repeat(255*gc_prob.detach().cpu().numpy()[:, None], 3, 1)
                        my_mesh = trimesh.Trimesh(vertices=smal_template_verts, faces=smal_faces, process=False,  maintain_order=True)
                        my_mesh.visual.vertex_colors = vert_colors
                        save_gc_mesh = True # False
                        if save_gc_mesh:
                            my_mesh.export(out_path_gcmesh)
            
                        '''
                        input_image = input[ind_img, :, :, :].detach().clone()
                        for t, m, s in zip(input_image, data_info.rgb_mean,data_info.rgb_stddev): t.add_(m)
                        input_image_np = input_image.detach().cpu().numpy().transpose(1, 2, 0) 
                        out_path = save_debug_path + 'b' + str(ind_img) +'_input.png'
                        plt.imsave(out_path, input_image_np)
                        '''

                        # -------------------------------------

                        # import pdb; pdb.set_trace()


                        '''
                        target_gc_class = target_dict['gc'][ind_img, :, 0]

                        current_vertices_smal = vertices_smal[ind_img, :, :]

                        points_centroid, plane_normal, error = fit_plane(current_vertices_smal[target_gc_class==1, :])
                        '''

                        # calculate ground plane
                        #   (see /is/cluster/work/nrueegg/icon_pifu_related/ICON/debug_code/curve_fitting_v2.py)
                        if return_mesh_with_gt_groundplane and 'gc' in target_dict.keys():

                            current_verts_remeshed = verts_remeshed[ind_img, :, :]
                            current_target_gc_class_remeshed_prep = target_gc_class_remeshed_prep[ind_img, ...]

                            if current_target_gc_class_remeshed_prep.sum() > 3:
                                points_on_plane = current_verts_remeshed[current_target_gc_class_remeshed_prep==1, :]
                                data_centroid, plane_normal, error = fit_plane(points_on_plane)
                                nonplane_points_centered = current_verts_remeshed[current_target_gc_class_remeshed_prep==0, :] - data_centroid[None, :]
                                nonplane_points_projected = torch.matmul(plane_normal[None, :], nonplane_points_centered.transpose(0,1))

                                if nonplane_points_projected.sum() > 0: # plane normal points towards the animal
                                    plane_normal = plane_normal.detach().cpu().numpy()
                                else:
                                    plane_normal = - plane_normal.detach().cpu().numpy()
                                data_centroid = data_centroid.detach().cpu().numpy()



                                # import pdb; pdb.set_trace()


                                desired_plane_normal_vector = np.asarray([[0, -1, 0]])
                                # new approach: use cross product
                                rotation_axis = np.cross(plane_normal, desired_plane_normal_vector)     #  np.cross(plane_normal, desired_plane_normal_vector)
                                lengt_rotation_axis = np.linalg.norm(rotation_axis)     # = sin(alpha)      (because vectors have unit length)
                                angle = np.sin(lengt_rotation_axis)
                                rot = R_sc.from_rotvec(angle * rotation_axis * 1/lengt_rotation_axis)
                                rot_mat = rot[0].as_matrix()
                                rot_upsidedown = R_sc.from_rotvec(np.pi * np.asarray([[1, 0, 0]]))
                                # rot_upsidedown[0].apply(rot[0].apply(plane_normal))
                                current_vertices_smal = vertices_smal[ind_img, :, :].detach().cpu().numpy()
                                new_smal_vertices = rot_upsidedown[0].apply(rot[0].apply(current_vertices_smal - data_centroid[None, :]))
                                my_mesh = trimesh.Trimesh(vertices=new_smal_vertices, faces=smal_faces, process=False,  maintain_order=True)
                                vert_colors[:, 2] = 255
                                my_mesh.visual.vertex_colors = vert_colors
                                out_path_gc_rotated = save_imgs_path + '/' + prefix + 'gc_rotated_' + img_name + '_new.obj'
                                my_mesh.export(out_path_gc_rotated)






                                '''# rot = R_sc.align_vectors(plane_normal.reshape((1, -1)), desired_plane_normal_vector)
                                desired_plane_normal_vector = np.asarray([[0, 1, 0]])

                                rot = R_sc.align_vectors(desired_plane_normal_vector, plane_normal.reshape((1, -1)))        # inv
                                rot_mat = rot[0].as_matrix()


                                current_vertices_smal = vertices_smal[ind_img, :, :].detach().cpu().numpy()
                                new_smal_vertices = rot[0].apply((current_vertices_smal - data_centroid[None, :]))

                                my_mesh = trimesh.Trimesh(vertices=new_smal_vertices, faces=smal_faces, process=False,  maintain_order=True)
                                my_mesh.visual.vertex_colors = vert_colors
                                out_path_gc_rotated = save_imgs_path + '/' + prefix + 'gc_rotated_' + img_name + '_y.obj'
                                my_mesh.export(out_path_gc_rotated)
                                '''









                        # ----


                        # -------------------------------------




            if index == 0:
                if len_dataset is None:
                    len_data = val_loader.batch_size * len(val_loader)  # 1703
                else:
                    len_data = len_dataset
                if metrics == 'all' or metrics == 'no_loss':
                    if result_network == 'normal':
                        summaries = {'normal': dict(), 'ref': dict()}
                        summary = summaries['normal']
                    else:
                        summary = summaries['ref']
                    summary['pck'] = np.zeros((len_data))
                    summary['pck_by_part'] = {group:np.zeros((len_data)) for group in KEYPOINT_GROUPS}
                    summary['acc_sil_2d'] = np.zeros(len_data)
                    summary['betas'] = np.zeros((len_data,betas.shape[1]))
                    summary['betas_limbs'] = np.zeros((len_data, betas_limbs.shape[1]))
                    summary['z'] = np.zeros((len_data, zz.shape[1]))
                    summary['pose_rotmat'] = np.zeros((len_data, pose_rotmat.shape[1], 3, 3))
                    summary['flength'] = np.zeros((len_data, flength.shape[1]))
                    summary['trans'] = np.zeros((len_data, trans.shape[1]))
                    summary['breed_indices'] = np.zeros((len_data))
                    summary['image_names'] = []        # len_data * [None]
            else:
                if result_network == 'normal':
                    summary = summaries['normal']
                else:
                    summary = summaries['ref']

            if save_imgs_path is not None:
                eval_save_visualizations_and_meshes(model, input, data_info, target_dict, test_name_list, vertices_smal, hg_keyp_norm, hg_keyp_scores, zz, betas, betas_limbs, pose_rotmat, trans, flength, pred_keyp, pred_silh, save_imgs_path, prefix, index, render_all=render_all)

            if metrics == 'all' or metrics == 'no_loss':
                preds = eval_prepare_pck_and_iou(model, input, data_info, target_dict, test_name_list, vertices_smal, hg_keyp_norm, hg_keyp_scores, zz, betas, betas_limbs, pose_rotmat, trans, flength, pred_keyp, pred_silh, save_imgs_path, prefix, index, pck_thresh, progress=progress)
                # add results for all images in this batch to lists
                curr_batch_size = pred_keyp.shape[0]
                eval_add_preds_to_summary(summary, preds, my_step, batch_size, curr_batch_size)
            else:
                # measure accuracy and record loss
                bs_fake = 1     # batch_size
        # import pdb; pdb.set_trace()


        # save_imgs_path + '/' + prefix + 'rot_tex_pred_' + img_name + '.png'
        # import pdb; pdb.set_trace()
        '''
        for ind_img in range(len(target_dict['index'])): 
            try: 
                if test_name_list is not None:
                    img_name = test_name_list[int(target_dict['index'][ind_img].cpu().detach().numpy())].replace('/', '_')
                    img_name = img_name.split('.')[0]
                else:
                    img_name = str(index) + '_' + str(ind_img)
                all_image_names = ['keypoints_pred_' + img_name + '.png',  'normal_comp_pred_' + img_name + '.png', 'normal_rot_tex_pred_' + img_name + '.png',  'ref_comp_pred_' + img_name + '.png', 'ref_rot_tex_pred_' + img_name + '.png']
                all_saved_images = []
                for sub_img_name in all_image_names:
                    saved_img = cv2.imread(save_imgs_path + '/' + sub_img_name)
                    if not (saved_img.shape[0] == 256 and saved_img.shape[1] == 256):
                        saved_img = cv2.resize(saved_img, (256, 256)) 
                    all_saved_images.append(saved_img)
                final_image = np.concatenate(all_saved_images, axis=1)
                save_imgs_path_sum = save_imgs_path.replace('test_', 'summary_test_')
                if not os.path.exists(save_imgs_path_sum): os.makedirs(save_imgs_path_sum)
                final_image_path = save_imgs_path_sum +  '/summary_' + img_name + '.png'
                cv2.imwrite(final_image_path, final_image)
            except: 
                print('dont save a summary image')
        '''
                
        
        bs_fake = 1
        if metrics == 'all' or metrics == 'no_loss':
            # update progress bar
            if progress is not None:
                '''my_string = "PCK: {0:.2f}, IOU: {1:.2f}".format(
                    pck[:(my_step * batch_size + curr_batch_size)].mean(),
                    acc_sil_2d[:(my_step * batch_size + curr_batch_size)].mean())'''
                my_string = "normal_PCK: {0:.2f}, normal_IOU: {1:.2f}, ref_PCK: {2:.2f}, ref_IOU: {3:.2f}".format(
                    summaries['normal']['pck'][:(my_step * batch_size + curr_batch_size)].mean(),
                    summaries['normal']['acc_sil_2d'][:(my_step * batch_size + curr_batch_size)].mean(),
                    summaries['ref']['pck'][:(my_step * batch_size + curr_batch_size)].mean(),
                    summaries['ref']['acc_sil_2d'][:(my_step * batch_size + curr_batch_size)].mean())
                progress.set_postfix_str(my_string)
        else:
            losses.update(loss_dict['loss'] + loss_dict_ref['loss'], bs_fake)
            losses_keyp.update(loss_dict['loss_keyp_weighted'], bs_fake)
            losses_silh.update(loss_dict['loss_silh_weighted'], bs_fake)
            losses_shape.update(loss_dict['loss_shape_weighted'], bs_fake)
            losses_pose.update(loss_dict['loss_poseprior_weighted'], bs_fake)   
            losses_class.update(loss_dict['loss_class_weighted'], bs_fake)
            losses_breed.update(loss_dict['loss_breed_weighted'], bs_fake)
            losses_partseg.update(loss_dict['loss_partseg_weighted'], bs_fake)
            losses_ref_keyp.update(loss_dict_ref['keyp_ref'], bs_fake)
            losses_ref_silh.update(loss_dict_ref['silh_ref'], bs_fake)
            loss_ref_pose = 0
            for l_name in ['pose_legs_side', 'pose_legs_tors', 'pose_tail_side', 'pose_tail_tors', 'pose_spine_side', 'pose_spine_tors']: 
                loss_ref_pose += loss_dict_ref[l_name]
            losses_ref_pose.update(loss_ref_pose, bs_fake)
            loss_ref_reg = 0
            for l_name in ['reg_trans', 'reg_flength', 'reg_pose']:
                loss_ref_reg += loss_dict_ref[l_name]
            losses_ref_reg.update(loss_ref_reg, bs_fake)
            acc = - loss_dict['loss_keyp_weighted']     # this will be used to keep track of the 'best model'
            accuracies.update(acc, bs_fake)
            # Show losses as part of the progress bar.
            if progress is not None:
                my_string = 'Loss: {loss:0.4f}, loss_keyp: {loss_keyp:0.4f}, loss_silh: {loss_silh:0.4f}, loss_partseg: {loss_partseg:0.4f}, loss_shape: {loss_shape:0.4f}, loss_pose: {loss_pose:0.4f}, loss_class: {loss_class:0.4f}, loss_breed: {loss_breed:0.4f}, loss_ref_keyp: {loss_ref_keyp:0.4f}, loss_ref_silh: {loss_ref_silh:0.4f}, loss_ref_pose: {loss_ref_pose:0.4f}, loss_ref_reg: {loss_ref_reg:0.4f}'.format(
                    loss=losses.avg,
                    loss_keyp=losses_keyp.avg,
                    loss_silh=losses_silh.avg,
                    loss_shape=losses_shape.avg,
                    loss_pose=losses_pose.avg,
                    loss_class=losses_class.avg,
                    loss_breed=losses_breed.avg,
                    loss_partseg=losses_partseg.avg,
                    loss_ref_keyp=losses_ref_keyp.avg,
                    loss_ref_silh=losses_ref_silh.avg,
                    loss_ref_pose=losses_ref_pose.avg,
                    loss_ref_reg=losses_ref_reg.avg)
                my_string_short = 'Loss: {loss:0.4f}, loss_keyp: {loss_keyp:0.4f}, loss_silh: {loss_silh:0.4f}, loss_ref_keyp: {loss_ref_keyp:0.4f}, loss_ref_silh: {loss_ref_silh:0.4f}, loss_ref_pose: {loss_ref_pose:0.4f}, loss_ref_reg: {loss_ref_reg:0.4f}'.format(
                    loss=losses.avg,
                    loss_keyp=losses_keyp.avg,
                    loss_silh=losses_silh.avg,
                    loss_ref_keyp=losses_ref_keyp.avg,
                    loss_ref_silh=losses_ref_silh.avg,
                    loss_ref_pose=losses_ref_pose.avg,
                    loss_ref_reg=losses_ref_reg.avg)
                progress.set_postfix_str(my_string_short)
        my_step += 1
    if metrics == 'all':
        return my_string, summaries     # summary    
    elif metrics == 'no_loss':
        return my_string, np.average(np.asarray(summaries['ref']['acc_sil_2d']))     # np.average(np.asarray(summary['acc_sil_2d']))
    else:
        return my_string, accuracies.avg       


# ---------------------------------------------------------------------------------------------------------------------------
def do_visual_epoch(val_loader, model, device, data_info, flip=False, quiet=False, acc_joints=None, save_imgs_path=None, weight_dict=None, weight_dict_ref=None, metrics=None, val_opt='default', test_name_list=None, render_all=False, pck_thresh=0.15, return_results=False, len_dataset=None):
    if save_imgs_path is not None:
        pathlib.Path(save_imgs_path).mkdir(parents=True, exist_ok=True) 
    all_results = []

    # Put the model in evaluation mode.
    model.eval()

    iterable = enumerate(val_loader)

    # information for normalization 
    norm_dict = {
        'pose_rot6d_mean': torch.from_numpy(data_info.pose_rot6d_mean).float().to(device),
        'trans_mean': torch.from_numpy(data_info.trans_mean).float().to(device),
        'trans_std': torch.from_numpy(data_info.trans_std).float().to(device),
        'flength_mean': torch.from_numpy(data_info.flength_mean).float().to(device),
        'flength_std': torch.from_numpy(data_info.flength_std).float().to(device)}


    return_mesh_with_gt_groundplane = True
    if return_mesh_with_gt_groundplane:
        remeshing_path = '/is/cluster/work/nrueegg/icon_pifu_related/barc_for_bite/data/smal_data_remeshed/uniform_surface_sampling/my_smpl_39dogsnorm_Jr_4_dog_remesh4000_info.pkl'
        with open(remeshing_path, 'rb') as fp: 
            remeshing_dict = pkl.load(fp)
        remeshing_relevant_faces = torch.tensor(remeshing_dict['smal_faces'][remeshing_dict['faceid_closest']], dtype=torch.long, device=device)
        remeshing_relevant_barys = torch.tensor(remeshing_dict['barys_closest'], dtype=torch.float32, device=device)

    # from smal_pytorch.smal_model.smal_torch_new import SMAL
    print('start: load smal default model (barc), but only for vertices')
    smal = SMAL()
    print('end: load smal default model (barc), but only for vertices')
    smal_template_verts = smal.v_template.detach().cpu().numpy()
    smal_faces = smal.faces.detach().cpu().numpy()

    file_alignment_errors = open(save_imgs_path + '/a_ref_procrustes_alignmnet_errors.txt', 'a') # append mode
    file_alignment_errors.write(" -----------  start evaluation  ------------- \n ")

    csv_file_alignment_errors = open(save_imgs_path + '/a_ref_procrustes_alignmnet_errors.csv', 'w') # write mode
    fieldnames = ['name', 'error']
    writer = csv.DictWriter(csv_file_alignment_errors, fieldnames=fieldnames)
    writer.writeheader()

    my_step = 0
    for index, (input, target_dict) in iterable:
        batch_size = input.shape[0]
        input = input.float().to(device)
        partial_results = {}

        # ----------------------- do visualization step -----------------------
        with torch.no_grad():
            output, output_unnorm, output_reproj, output_ref, output_ref_comp = model(input, norm_dict=norm_dict)        


        # import pdb; pdb.set_trace()


        sm = torch.nn.Softmax(dim=2)
        ground_contact_probs = sm(output_ref['vertexwise_ground_contact'])

        for result_network in ['normal', 'ref']:
            # variabled that are not refined
            hg_keyp_norm = output['keypoints_norm']
            hg_keyp_scores = output['keypoints_scores']
            betas = output_reproj['betas']
            betas_limbs = output_reproj['betas_limbs']
            zz = output_reproj['z']
            if result_network == 'normal':
                # STEP 1: normal network
                vertices_smal = output_reproj['vertices_smal']
                flength = output_unnorm['flength']
                pose_rotmat = output_unnorm['pose_rotmat']
                trans = output_unnorm['trans']
                pred_keyp = output_reproj['keyp_2d']
                pred_silh = output_reproj['silh']
                prefix = 'normal_'
            else:
                # STEP 1: refinement network
                vertices_smal = output_ref['vertices_smal']
                flength = output_ref['flength']
                pose_rotmat = output_ref['pose_rotmat']
                trans = output_ref['trans']
                pred_keyp = output_ref['keyp_2d']
                pred_silh = output_ref['silh']
                prefix = 'ref_'
                
                bs = vertices_smal.shape[0]
                # target_gc_class = target_dict['gc'][:, :, 0]
                target_gc_class = torch.round(ground_contact_probs).long()[:, :, 1]
                sel_verts = torch.index_select(output_ref['vertices_smal'], dim=1, index=remeshing_relevant_faces.reshape((-1))).reshape((bs, remeshing_relevant_faces.shape[0], 3, 3))
                verts_remeshed = torch.einsum('ij,aijk->aik', remeshing_relevant_barys, sel_verts)
                target_gc_class_remeshed = torch.einsum('ij,aij->ai', remeshing_relevant_barys, target_gc_class[:, remeshing_relevant_faces].to(device=device, dtype=torch.float32))
                target_gc_class_remeshed_prep = torch.round(target_gc_class_remeshed).to(torch.long)




                # index = i  
                # ind_img = 0
                for ind_img in range(batch_size): #  range(min(12, batch_size)):     # range(12):    # [0]:  #range(0, batch_size):

                    # ind_img = 0
                    if test_name_list is not None:
                        img_name = test_name_list[int(target_dict['index'][ind_img].cpu().detach().numpy())].replace('/', '_')
                        img_name = img_name.split('.')[0]
                    else:
                        img_name = str(index) + '_' + str(ind_img)
                    out_path_gcmesh = save_imgs_path + '/' + prefix + 'gcmesh_' + img_name + '.obj'

                    gc_prob = ground_contact_probs[ind_img, :, 1]   # contact probability
                    vert_colors = np.repeat(255*gc_prob.detach().cpu().numpy()[:, None], 3, 1)
                    my_mesh = trimesh.Trimesh(vertices=smal_template_verts, faces=smal_faces, process=False,  maintain_order=True)
                    my_mesh.visual.vertex_colors = vert_colors
                    save_gc_mesh = False
                    if save_gc_mesh:
                        my_mesh.export(out_path_gcmesh)            
        
                    current_verts_remeshed = verts_remeshed[ind_img, :, :]
                    current_target_gc_class_remeshed_prep = target_gc_class_remeshed_prep[ind_img, ...]

                    if current_target_gc_class_remeshed_prep.sum() > 3:
                        points_on_plane = current_verts_remeshed[current_target_gc_class_remeshed_prep==1, :]
                        data_centroid, plane_normal, error = fit_plane(points_on_plane)
                        nonplane_points_centered = current_verts_remeshed[current_target_gc_class_remeshed_prep==0, :] - data_centroid[None, :]
                        nonplane_points_projected = torch.matmul(plane_normal[None, :], nonplane_points_centered.transpose(0,1))

                        if nonplane_points_projected.sum() > 0: # plane normal points towards the animal
                            plane_normal = plane_normal.detach().cpu().numpy()
                        else:
                            plane_normal = - plane_normal.detach().cpu().numpy()
                        data_centroid = data_centroid.detach().cpu().numpy()



                        # import pdb; pdb.set_trace()


                        desired_plane_normal_vector = np.asarray([[0, -1, 0]])
                        # new approach: use cross product
                        rotation_axis = np.cross(plane_normal, desired_plane_normal_vector)     #  np.cross(plane_normal, desired_plane_normal_vector)
                        lengt_rotation_axis = np.linalg.norm(rotation_axis)     # = sin(alpha)      (because vectors have unit length)
                        angle = np.sin(lengt_rotation_axis)
                        rot = R_sc.from_rotvec(angle * rotation_axis * 1/lengt_rotation_axis)
                        rot_mat = rot[0].as_matrix()
                        rot_upsidedown = R_sc.from_rotvec(np.pi * np.asarray([[1, 0, 0]]))
                        # rot_upsidedown[0].apply(rot[0].apply(plane_normal))
                        current_vertices_smal = vertices_smal[ind_img, :, :].detach().cpu().numpy()
                        new_smal_vertices = rot_upsidedown[0].apply(rot[0].apply(current_vertices_smal - data_centroid[None, :]))
                        my_mesh = trimesh.Trimesh(vertices=new_smal_vertices, faces=smal_faces, process=False,  maintain_order=True)
                        vert_colors[:, 2] = 255
                        my_mesh.visual.vertex_colors = vert_colors
                        out_path_gc_rotated = save_imgs_path + '/' + prefix + 'gc_rotated_' + img_name + '_new.obj'
                        my_mesh.export(out_path_gc_rotated)



                        '''
                        import pdb; pdb.set_trace()

                        from src.evaluation.registration import preprocess_point_cloud, o3d_ransac, draw_registration_result
                        import open3d as o3d
                        import copy


                        mesh_gt_path = target_dict['mesh_path'][ind_img]
                        mesh_gt = o3d.io.read_triangle_mesh(mesh_gt_path)

                        mesh_gt_verts = np.asarray(mesh_gt.vertices)
                        mesh_gt_faces = np.asarray(mesh_gt.triangles)
                        diag_gt = np.sqrt(sum((mesh_gt_verts.max(axis=0) - mesh_gt_verts.min(axis=0))**2))

                        mesh_pred_verts = np.asarray(new_smal_vertices)
                        mesh_pred_faces = np.asarray(smal_faces)
                        diag_pred = np.sqrt(sum((mesh_pred_verts.max(axis=0) - mesh_pred_verts.min(axis=0))**2))
                        mesh_pred = o3d.geometry.TriangleMesh()
                        mesh_pred.vertices = o3d.utility.Vector3dVector(mesh_pred_verts)
                        mesh_pred.triangles = o3d.utility.Vector3iVector(mesh_pred_faces)

                        # center the predicted mesh around 0
                        trans = - mesh_pred_verts.mean(axis=0)
                        mesh_pred_verts_new = mesh_pred_verts + trans
                        # change the size of the predicted mesh
                        mesh_pred_verts_new = mesh_pred_verts_new * diag_gt / diag_pred

                        # transform the predicted mesh (rough alignment)
                        mesh_pred_new = copy.deepcopy(mesh_pred)
                        mesh_pred_new.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_pred_verts_new))    # normals should not have changed
                        voxel_size = 0.01       # 0.5
                        distance_threshold = 0.015  # 0.005 #  0.02   # 1.0
                        result, src_down, src_fpfh, dst_down, dst_fpfh = o3d_ransac(mesh_pred_new, mesh_gt, voxel_size=voxel_size, distance_threshold=distance_threshold, return_all=True)
                        transform = result.transformation
                        mesh_pred_transf = copy.deepcopy(mesh_pred_new).transform(transform)

                        out_path_pred_transf = save_imgs_path + '/' + prefix + 'alignment_initial_' + img_name + '.obj'
                        o3d.io.write_triangle_mesh(out_path_pred_transf, mesh_pred_transf)

                        # img_name_part = img_name.split(img_name.split('_')[-1] + '_')[0]
                        # out_path_gt = save_imgs_path + '/' + prefix + 'ground_truth_' + img_name_part + '.obj'
                        # o3d.io.write_triangle_mesh(out_path_gt, mesh_gt)

                        
                        trans_init = transform
                        threshold = 0.02        #  0.1  # 0.02

                        n_points = 10000
                        src = mesh_pred_new.sample_points_uniformly(number_of_points=n_points)
                        dst = mesh_gt.sample_points_uniformly(number_of_points=n_points)

                        # reg_p2p = o3d.pipelines.registration.registration_icp(src_down, dst_down, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
                        reg_p2p = o3d.pipelines.registration.registration_icp(src, dst, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

                        # mesh_pred_transf_refined = copy.deepcopy(mesh_pred_new).transform(reg_p2p.transformation)
                        # out_path_pred_transf_refined =  save_imgs_path + '/' + prefix + 'alignment_final_' + img_name + '.obj'
                        # o3d.io.write_triangle_mesh(out_path_pred_transf_refined, mesh_pred_transf_refined)


                        aligned_mesh_final = trimesh.Trimesh(mesh_pred_new.vertices, mesh_pred_new.triangles, vertex_colors=[0, 255, 0])
                        gt_mesh = trimesh.Trimesh(mesh_gt.vertices, mesh_gt.triangles, vertex_colors=[255, 0, 0])
                        scene = trimesh.Scene([aligned_mesh_final, gt_mesh])
                        out_path_alignment_with_gt =  save_imgs_path + '/' + prefix + 'alignment_with_gt_' + img_name + '.obj'

                        scene.export(out_path_alignment_with_gt)
                        '''

                        # import pdb; pdb.set_trace()


                        # SMAL_KEYPOINT_NAMES_FOR_3D_EVAL     # 17 keypoints
                        # prepare target
                        target_keyp_isvalid = target_dict['keypoints_3d'][ind_img, :, 3].detach().cpu().numpy()
                        keyp_to_use = (np.asarray(SMAL_KEYPOINT_WHICHTOUSE_FOR_3D_EVAL)==1)*(target_keyp_isvalid==1)
                        target_keyp_raw = target_dict['keypoints_3d'][ind_img, :, :3].detach().cpu().numpy()
                        target_keypoints = target_keyp_raw[keyp_to_use, :]
                        target_pointcloud = target_dict['pointcloud_points'][ind_img, :, :].detach().cpu().numpy()
                        # prepare prediction
                        pred_keypoints_raw = output_ref['vertices_smal'][ind_img, SMAL_KEYPOINT_INDICES_FOR_3D_EVAL, :].detach().cpu().numpy()
                        pred_keypoints = pred_keypoints_raw[keyp_to_use, :]
                        pred_pointcloud =  verts_remeshed[ind_img, :, :].detach().cpu().numpy()




                        '''
                        pred_keypoints_transf, pred_pointcloud_transf, procrustes_params = compute_similarity_transform(pred_keypoints, target_keypoints, num_joints=None, verts=pred_pointcloud)
                        pa_error = np.sqrt(np.sum((target_keypoints - pred_keypoints_transf) ** 2, axis=1))
                        error_procrustes = np.mean(pa_error)


                        col_target = np.zeros((target_pointcloud.shape[0], 3), dtype=np.uint8)
                        col_target[:, 0] = 255
                        col_pred = np.zeros((pred_pointcloud_transf.shape[0], 3), dtype=np.uint8)
                        col_pred[:, 1] = 255
                        pc = trimesh.points.PointCloud(np.concatenate((target_pointcloud, pred_pointcloud_transf)), colors=np.concatenate((col_target, col_pred)))
                        out_path_pc = save_imgs_path + '/' + prefix + 'pointclouds_aligned_' + img_name + '.obj'
                        pc.export(out_path_pc)

                        print(target_dict['mesh_path'][ind_img])
                        print(error_procrustes)
                        file_alignment_errors.write(target_dict['mesh_path'][ind_img] + '\n')
                        file_alignment_errors.write('error: ' + str(error_procrustes) + ' \n')

                        writer.writerow({'name': (target_dict['mesh_path'][ind_img]).split('/')[-1], 'error': str(error_procrustes)})

                        # import pdb; pdb.set_trace()
                        # alignment_dict = calculate_alignemnt_errors(output_ref['vertices_smal'][ind_img, :, :], target_dict['keypoints_3d'][ind_img, :, :], target_dict['pointcloud_points'][ind_img, :, :])
                        # file_alignment_errors.write('error: ' + str(alignment_dict['error_procrustes']) + ' \n')
                        '''






            if index == 0:
                if len_dataset is None:
                    len_data = val_loader.batch_size * len(val_loader)  # 1703
                else:
                    len_data = len_dataset
                if result_network == 'normal':
                    summaries = {'normal': dict(), 'ref': dict()}
                    summary = summaries['normal']
                else:
                    summary = summaries['ref']
                summary['pck'] = np.zeros((len_data))
                summary['pck_by_part'] = {group:np.zeros((len_data)) for group in KEYPOINT_GROUPS}
                summary['acc_sil_2d'] = np.zeros(len_data)
                summary['betas'] = np.zeros((len_data,betas.shape[1]))
                summary['betas_limbs'] = np.zeros((len_data, betas_limbs.shape[1]))
                summary['z'] = np.zeros((len_data, zz.shape[1]))
                summary['pose_rotmat'] = np.zeros((len_data, pose_rotmat.shape[1], 3, 3))
                summary['flength'] = np.zeros((len_data, flength.shape[1]))
                summary['trans'] = np.zeros((len_data, trans.shape[1]))
                summary['breed_indices'] = np.zeros((len_data))
                summary['image_names'] = []        # len_data * [None]
                # ['vertices_smal'] = np.zeros((len_data, vertices_smal.shape[1], 3))
            else:
                if result_network == 'normal':
                    summary = summaries['normal']
                else:
                    summary = summaries['ref']
            
            
            # import pdb; pdb.set_trace()


            eval_save_visualizations_and_meshes(model, input, data_info, target_dict, test_name_list, vertices_smal, hg_keyp_norm, hg_keyp_scores, zz, betas, betas_limbs, pose_rotmat, trans, flength, pred_keyp, pred_silh, save_imgs_path, prefix, index, render_all=render_all)


            preds = eval_prepare_pck_and_iou(model, input, data_info, target_dict, test_name_list, vertices_smal, hg_keyp_norm, hg_keyp_scores, zz, betas, betas_limbs, pose_rotmat, trans, flength, pred_keyp, pred_silh, save_imgs_path, prefix, index, pck_thresh=None, skip_pck_and_iou=True)
            # add results for all images in this batch to lists
            curr_batch_size = pred_keyp.shape[0]
            eval_add_preds_to_summary(summary, preds, my_step, batch_size, curr_batch_size, skip_pck_and_iou=True)

            # summary['vertices_smal'][my_step * batch_size:my_step * batch_size + curr_batch_size] = vertices_smal.detach().cpu().numpy()



            
            
            
            
            
            
            
            
            
            
            
            
            '''
            try: 
                if test_name_list is not None:
                    img_name = test_name_list[int(target_dict['index'][ind_img].cpu().detach().numpy())].replace('/', '_')
                    img_name = img_name.split('.')[0]
                else:
                    img_name = str(index) + '_' + str(ind_img)
                partial_results['img_name'] = img_name
                visualizations = model.render_vis_nograd(vertices=output_reproj['vertices_smal'],
                                                        focal_lengths=output_unnorm['flength'],
                                                        color=0)    # 2)
                # save image with predicted keypoints
                pred_unp = (output['keypoints_norm'][ind_img, :, :] + 1.) / 2 * (data_info.image_size - 1)
                pred_unp_maxval = output['keypoints_scores'][ind_img, :, :]
                pred_unp_prep = torch.cat((pred_unp, pred_unp_maxval), 1)
                inp_img = input[ind_img, :, :, :].detach().clone()
                if save_imgs_path is not None:
                    out_path = save_imgs_path + '/keypoints_pred_' + img_name + '.png'
                    save_input_image_with_keypoints(inp_img, pred_unp_prep, out_path=out_path, threshold=0.1, print_scores=True, ratio_in_out=1.0)    # threshold=0.3
                # save predicted 3d model
                #   (1) front view
                pred_tex = visualizations[ind_img, :, :, :].permute((1, 2, 0)).cpu().detach().numpy() / 256
                pred_tex_max = np.max(pred_tex, axis=2)
                partial_results['tex_pred'] = pred_tex
                if save_imgs_path is not None:
                    out_path = save_imgs_path + '/tex_pred_' + img_name + '.png'
                    plt.imsave(out_path, pred_tex)
                input_image = input[ind_img, :, :, :].detach().clone()
                for t, m, s in zip(input_image, data_info.rgb_mean, data_info.rgb_stddev): t.add_(m)
                input_image_np = input_image.detach().cpu().numpy().transpose(1, 2, 0) 
                im_masked = cv2.addWeighted(input_image_np,0.2,pred_tex,0.8,0)
                im_masked[pred_tex_max<0.01, :] = input_image_np[pred_tex_max<0.01, :]
                partial_results['comp_pred'] = im_masked
                if save_imgs_path is not None:
                    out_path = save_imgs_path + '/comp_pred_' + img_name + '.png'
                    plt.imsave(out_path, im_masked)
                #   (2) side view
                vertices_cent = output_reproj['vertices_smal'] - output_reproj['vertices_smal'].mean(dim=1)[:, None, :]
                roll = np.pi / 2 * torch.ones(1).float().to(device)
                pitch = np.pi / 2 * torch.ones(1).float().to(device)
                tensor_0 = torch.zeros(1).float().to(device)
                tensor_1 = torch.ones(1).float().to(device)
                RX = torch.stack([torch.stack([tensor_1, tensor_0, tensor_0]), torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3,3)
                RY = torch.stack([
                    torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3,3)
                vertices_rot = (torch.matmul(RY, vertices_cent.reshape((-1, 3))[:, :, None])).reshape((batch_size, -1, 3))
                vertices_rot[:, :, 2] = vertices_rot[:, :, 2] + torch.ones_like(vertices_rot[:, :, 2]) * 20     # 18     # *16
                visualizations_rot = model.render_vis_nograd(vertices=vertices_rot,
                                                        focal_lengths=output_unnorm['flength'],
                                                        color=0)    # 2)
                pred_tex = visualizations_rot[ind_img, :, :, :].permute((1, 2, 0)).cpu().detach().numpy() / 256
                pred_tex_max = np.max(pred_tex, axis=2)
                partial_results['rot_tex_pred'] = pred_tex
                if save_imgs_path is not None:
                    out_path = save_imgs_path + '/rot_tex_pred_' + img_name + '.png'
                    plt.imsave(out_path, pred_tex)
                render_all = True
                if render_all:
                    # save input image 
                    inp_img = input[ind_img, :, :, :].detach().clone()
                    if save_imgs_path is not None:
                        out_path = save_imgs_path + '/image_' + img_name + '.png'
                        save_input_image(inp_img, out_path)
                    # save posed mesh
                    V_posed = output_reproj['vertices_smal'][ind_img, :, :].detach().cpu().numpy()
                    Faces = model.smal.f
                    mesh_posed = trimesh.Trimesh(vertices=V_posed, faces=Faces, process=False,  maintain_order=True)
                    partial_results['mesh_posed'] = mesh_posed
                    if save_imgs_path is not None:
                        mesh_posed.export(save_imgs_path + '/mesh_posed_' + img_name + '.obj')
            except:
                print('pass...')
            all_results.append(partial_results)
            '''

        my_step += 1


    file_alignment_errors.close()
    csv_file_alignment_errors.close()


    if return_results:
        return all_results
    else:
        return summaries