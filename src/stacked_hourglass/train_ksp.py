
'''
in contrast to train.py, here we do not only predict keypoints but instead:
    - keypoints
    - segmentation
'''

import torch
import torch.backends.cudnn
import torch.nn.parallel
import torch.nn as nn
from tqdm import tqdm
import os
import pathlib
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pickle as pkl

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# from stacked_hourglass.loss import joints_mse_loss
from stacked_hourglass.loss import joints_mse_loss_onKPloc, segmentation_loss
from stacked_hourglass.utils.evaluation import accuracy, AverageMeter, final_preds, get_preds, get_preds_soft
from stacked_hourglass.utils.transforms import fliplr, flip_back
from stacked_hourglass.utils.visualization import save_input_image_with_keypoints, save_image_with_part_segmentation, save_image_with_part_segmentation_from_gt_annotation



def do_training_step(model, optimiser, input, target, meta, data_info, target_weight=None):
    assert model.training, 'model must be in training mode.'
    assert len(input) == len(target), 'input and target must contain the same number of examples.'

    with torch.enable_grad():

        # Forward pass and loss calculation.
        out_dict = model(input)    

        # original: loss = sum(joints_mse_loss(o, target, target_weight) for o in output)
        '''loss_kp = sum(joints_mse_loss_onKPloc(o[:, :-2, :, :], target, meta, target_weight) for o in output)
        loss_seg = sum(segmentation_loss(o[:, -2:, :, :], meta) for o in output)'''
        loss_kp = sum(joints_mse_loss_onKPloc(o, target, meta, target_weight) for o in out_dict['out_list_kp'])
        loss_seg = sum(segmentation_loss(o, meta) for o in out_dict['out_list_seg'])
        loss_seg_big = segmentation_loss(out_dict['seg_final'], meta)

        # for the second stage where we add a dataset with body part segmentations
        #   and not just fake -1 labels, we calculate body part segmentation loss as well
        # if all body part labels are -1, we ignore this loss calculation
        if meta['body_part_matrix'].max() > -1:     # this will be the case for dogsvoc but not stanext 
            tbp_dict = {'full_body': [0, 8], 
                        'head': [8, 13], 
                        'torso': [13, 15]}
            loss_partseg = []
            criterion_ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
            ''''weights = [5.0, 1.0, 1.0, 1.0, 1.0]
            class_weights = torch.FloatTensor(weights).to(input.device)
            criterion_ce_weighted = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1, weight=class_weights)
            for ind_tbp, part in enumerate(['full_body', 'head', 'torso']):
                tbp_out = out_dict['partseg_final'][:, tbp_dict[part][0]:tbp_dict[part][1], :, :]
                tbp_target = meta['body_part_matrix'][:, ind_tbp, :, :].to(torch.long)
                if part == 'head':
                    loss_partseg.append(criterion_ce_weighted(tbp_out, tbp_target))
                else:
                    loss_partseg.append(criterion_ce(tbp_out, tbp_target))'''
            for ind_tbp, part in enumerate(['full_body', 'head', 'torso']):
                tbp_out = out_dict['partseg_final'][:, tbp_dict[part][0]:tbp_dict[part][1], :, :]
                tbp_target = meta['body_part_matrix'][:, ind_tbp, :, :].to(torch.long)
                if part == 'full_body':
                    # ignore parts of the silhouette which dont have a specific body part label
                    tbp_target[tbp_target==0] = -1  
                    loss_partseg.append(criterion_ce(tbp_out, tbp_target))
                else:
                    loss_partseg.append(criterion_ce(tbp_out, tbp_target))

            # loss = loss_kp + loss_seg*0.01 + loss_seg_big*0.1       # orig     # 0.001       # 0.01
            loss = loss_kp + loss_seg*0.001 + loss_seg_big*0.01 + 0.01*(loss_partseg[0] + loss_partseg[1] + loss_partseg[2])

        else:
            loss = loss_kp + loss_seg*0.01 + loss_seg_big*0.1

        # Backward pass and parameter update.
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        loss_dict = {'loss': loss.item(),
                    'keyp': loss_kp.item(),
                    'seg': loss_seg.item(),
                    'seg_big': loss_seg_big.item()
                    }

    return out_dict['out_list_kp'][-1], loss_dict


def do_training_epoch(train_loader, model, device, data_info, optimiser, quiet=False, acc_joints=None):
    losses = AverageMeter()
    accuracies = AverageMeter()

    # Put the model in training mode.
    model.train()

    iterable = enumerate(train_loader)
    progress = None
    if not quiet:
        progress = tqdm(iterable, desc='Train', total=len(train_loader), ascii=True, leave=False)
        iterable = progress

    for i, (input, target, meta) in iterable:
        input, target = input.to(device), target.to(device, non_blocking=True)
        target_weight = meta['target_weight'].to(device, non_blocking=True)
        meta['silh'] = meta['silh'].to(device, non_blocking=True)
        meta['body_part_matrix'] = meta['body_part_matrix'].to(device, non_blocking=True)

        output_kp, loss_dict = do_training_step(model, optimiser, input, target, meta, data_info, target_weight)
        loss = loss_dict['loss']

        acc = accuracy(output_kp, target, acc_joints)

        # measure accuracy and record loss
        losses.update(loss, input.size(0))
        accuracies.update(acc[0], input.size(0))

        # Show accuracy and loss as part of the progress bar.
        if progress is not None:
            progress.set_postfix_str('Loss: {loss:0.4f}, Acc: {acc:6.2f}'.format(
                loss=losses.avg,
                acc=100 * accuracies.avg
            ))

    return losses.avg, accuracies.avg


def do_validation_step(model, input, target, meta, data_info, target_weight=None, flip=False):
    assert not model.training, 'model must be in evaluation mode.'
    assert len(input) == len(target), 'input and target must contain the same number of examples.'

    # Forward pass and loss calculation.
    # output = model(input)
    out_dict = model(input)     # ['out_list', 'seg_final']
    '''output = out_dict['out_list']'''

    # original: loss = sum(joints_mse_loss(o, target, target_weight) for o in output)
    '''loss_kp = sum(joints_mse_loss_onKPloc(o[:, :-2, :, :], target, meta, target_weight) for o in output)
    loss_seg = sum(segmentation_loss(o[:, -2:, :, :], meta) for o in output)'''
    loss_kp = sum(joints_mse_loss_onKPloc(o, target, meta, target_weight) for o in out_dict['out_list_kp'])
    loss_seg = sum(segmentation_loss(o, meta) for o in out_dict['out_list_seg'])
    loss_seg_big = segmentation_loss(out_dict['seg_final'], meta)
    loss = loss_kp + loss_seg*0.01 + loss_seg_big*0.1            # 0.001       # 0.01

    # Get the heatmaps.
    heatmaps = out_dict['out_list_kp'][-1].cpu()

    '''seg = output[-1][:, -2:, :, :].cpu()'''
    seg = out_dict['out_list_seg'][-1].cpu()
    seg_big = out_dict['seg_final'].cpu()
    partseg_big = out_dict['partseg_final'].cpu()

    loss_dict = {'loss': loss.item(),
                'keyp': loss_kp.item(),
                'seg': loss_seg.item(),
                'seg_big': loss_seg_big.item()
                }

    return heatmaps, seg, seg_big, partseg_big, loss_dict      # loss.item()


def do_validation_epoch(val_loader, model, device, data_info, flip=False, quiet=False, acc_joints=None, save_imgs_path=None, save_pkl_path=None):
    losses = AverageMeter()
    accuracies = AverageMeter()
    predictions = [None] * len(val_loader.dataset)

    if save_imgs_path is not None:
        pathlib.Path(save_imgs_path).mkdir(parents=True, exist_ok=True) 

    # Put the model in evaluation mode.
    model.eval()

    iterable = enumerate(val_loader)
    progress = None
    if not quiet:
        progress = tqdm(iterable, desc='Valid', total=len(val_loader), ascii=True, leave=False)
        iterable = progress

    for i, (input, target, meta) in iterable:
        # Copy data to the training device (eg GPU).
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_weight = meta['target_weight'].to(device, non_blocking=True)
        meta['silh'] = meta['silh'].to(device, non_blocking=True)
        if 'body_part_matrix' in meta.keys():
            meta['body_part_matrix'] = meta['body_part_matrix'].to(device, non_blocking=True)

        heatmaps, seg, seg_big, partseg_big, loss_dict = do_validation_step(model, input, target, meta, data_info, target_weight, flip)
        loss = loss_dict['loss']

        # Calculate PCK from the predicted heatmaps.
        acc = accuracy(heatmaps, target.cpu(), acc_joints)

        # Calculate locations in original image space from the predicted heatmaps.
        preds = final_preds(heatmaps, meta['center'], meta['scale'], [64, 64])
        # NEW for visualization: (and redundant, but for visualization)
        if (save_imgs_path is not None) or (save_pkl_path is not None):
            preds_unprocessed, preds_unprocessed_norm, preds_unprocessed_maxval = get_preds_soft(heatmaps, return_maxval=True, norm_and_unnorm_coords=True)

        ind = 0
        for example_index, pose in zip(meta['index'], preds):
            # prepare save paths
            if save_pkl_path is not None:
                out_name_seg_overlay = os.path.join(save_imgs_path, meta['name'][ind].replace('.jpg', '__') + 'seg_overlay.png')
                out_name_kp = os.path.join(save_imgs_path, meta['name'][ind].replace('.jpg', '__') + 'res.png')
                if not os.path.exists(os.path.dirname(out_name_kp)):
                    os.makedirs(os.path.dirname(out_name_kp))
                out_name_pkl = os.path.join(save_pkl_path, meta['name'][ind].replace('.jpg', '.pkl'))
                if not os.path.exists(os.path.dirname(out_name_pkl)):
                    os.makedirs(os.path.dirname(out_name_pkl))
            else:
                if save_imgs_path is not None:
                    out_name_seg_overlay = os.path.join(save_imgs_path, 'seg_overlay_' + str( example_index.item()) + '.png')
                    out_name_kp = os.path.join(save_imgs_path, 'res_' + str( example_index.item()) + '.png')
            predictions[example_index] = pose
            # NEW for visualization
            if save_imgs_path is not None:
                soft_max = torch.nn.Softmax(dim= 0)
                segm_img_pred = soft_max((seg_big[ind, :, :, :]))[1, :, :]
                if save_pkl_path is None:
                    # save segmentation image
                    out_name_seg = os.path.join(save_imgs_path, 'seg_' + str( example_index.item()) + '.png')
                    segm_img_pred_small = soft_max((seg[ind, :, :, :]))[1, :, :]
                    plt.imsave(out_name_seg, segm_img_pred_small)
                    # save segmentation image
                    out_name_seg = os.path.join(save_imgs_path, 'seg_big_' + str( example_index.item()) + '.png')
                    plt.imsave(out_name_seg, segm_img_pred)
                # segmentation overlay
                input_image = input[ind, :, :, :].detach().clone()
                for t, m, s in zip(input_image, data_info.rgb_mean, data_info.rgb_stddev): t.add_(m)
                input_image_np = input_image.detach().cpu().numpy().transpose(1, 2, 0) 
                thr = 0.3
                segm_img_pred[segm_img_pred>thr] = 1
                segm_img_pred_3 = np.stack([segm_img_pred, np.zeros((256, 256), dtype=np.float32), np.zeros((256, 256), dtype=np.float32)], axis=2)
                segm_img_pred_3[segm_img_pred<thr] = input_image_np[segm_img_pred<thr]
                im_masked = cv2.addWeighted(input_image_np,0.5,segm_img_pred_3,0.5,0)
                plt.imsave(out_name_seg_overlay, im_masked)
                # save keypoint image
                pred_unp = preds_unprocessed[ind, :, :]
                pred_unp_maxval = preds_unprocessed_maxval[ind, :, :]
                pred_unp_prep = torch.cat((pred_unp, pred_unp_maxval), 1)
                inp_img = input[ind, :, :, :]
                save_input_image_with_keypoints(inp_img, pred_unp_prep, out_path=out_name_kp, threshold=0.1, print_scores=True)    # threshold=0.3

                # NEW: save body part segmentation image:
                out_path_seg = os.path.join(save_imgs_path, 'partseg_overlay_' + str( example_index.item()) + '.png')
                out_path_seg_overlay = os.path.join(save_imgs_path, 'partseg_overlay2_' + str( example_index.item()) + '.png')
                save_image_with_part_segmentation(partseg_big, seg_big, input_image_np, ind, out_path_seg, out_path_seg_overlay, thr=thr)

                # save pkl with results
                if save_pkl_path is not None:
                    result_dict = {'keypoints_normalized': preds_unprocessed_norm[ind, :, :],
                                    'keypoints_scores': preds_unprocessed_maxval[ind, :, :],
                                    'segmentation_fg': soft_max((seg_big[ind, :, :, :]))[1, :, :],
                                    'augmentation_corrected': False,
                                    'center': meta['center'],
                                    'scale': meta['scale'],
                                    'resolution': meta['resolution'],
                                    'do_flip': meta['do_flip'],
                                    'rot': meta['rot']}
                    with open(out_name_pkl, 'wb') as handle:
                        pkl.dump(result_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
            ind += 1

        # Record accuracy and loss for this batch.
        losses.update(loss, input.size(0))
        accuracies.update(acc[0].item(), input.size(0))

        # Show accuracy and loss as part of the progress bar.
        if progress is not None:
            progress.set_postfix_str('Loss: {loss:0.4f}, Acc: {acc:6.2f}'.format(
                loss=losses.avg,
                acc=100 * accuracies.avg
            ))

    predictions = torch.stack(predictions, dim=0)

    return losses.avg, accuracies.avg, predictions
