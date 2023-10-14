
# scripts/train.py --workers 12 --checkpoint project22_no3dcgloss_smaldogsilvia_v0 --loss-weight-path barc_loss_weights_no3dcgloss.json --config barc_cfg_train.yaml start --model-file-hg hg_ksp_fromnewanipose_stanext_v0/checkpoint.pth.tar --model-file-3d barc_normflow_pret/checkpoint.pth.tar

import torch
import torch.backends.cudnn
import torch.nn.parallel
from tqdm import tqdm
import os
import json
import pathlib

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../', 'src'))
# from stacked_hourglass.loss import joints_mse_loss
from stacked_hourglass.loss import joints_mse_loss_onKPloc
from stacked_hourglass.utils.evaluation import accuracy, AverageMeter, final_preds, get_preds, get_preds_soft
from stacked_hourglass.utils.transforms import fliplr, flip_back
from stacked_hourglass.utils.visualization import save_input_image_with_keypoints


def do_training_step(model, optimiser, input, target, meta, data_info, target_weight=None):
    assert model.training, 'model must be in training mode.'
    assert len(input) == len(target), 'input and target must contain the same number of examples.'

    with torch.enable_grad():
        # Forward pass and loss calculation.
        output = model(input)

        # original: loss = sum(joints_mse_loss(o, target, target_weight) for o in output)
        loss = sum(joints_mse_loss_onKPloc(o, target, meta, target_weight) for o in output)

        # Backward pass and parameter update.
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    return output[-1], loss.item()


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

        output, loss = do_training_step(model, optimiser, input, target, meta, data_info, target_weight)

        acc = accuracy(output, target, acc_joints)

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
    # assert not model.training, 'model must be in evaluation mode.'
    assert len(input) == len(target), 'input and target must contain the same number of examples.'

    # Forward pass and loss calculation.
    output = model(input)

    # original: loss = sum(joints_mse_loss(o, target, target_weight) for o in output)
    loss = sum(joints_mse_loss_onKPloc(o, target, meta, target_weight) for o in output)

    # Get the heatmaps.
    if flip:
        # If `flip` is true, perform horizontally flipped inference as well. This should
        # result in more robust predictions at the expense of additional compute.
        flip_input = fliplr(input)
        flip_output = model(flip_input)
        flip_output = flip_output[-1].cpu()
        flip_output = flip_back(flip_output.detach(), data_info.hflip_indices)
        heatmaps = (output[-1].cpu() + flip_output) / 2
    else:
        heatmaps = output[-1].cpu()


    return heatmaps, loss.item()


def do_validation_epoch(val_loader, model, device, data_info, flip=False, quiet=False, acc_joints=None, save_imgs_path=None):
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
        heatmaps, loss = do_validation_step(model, input, target, meta, data_info, target_weight, flip)

        # Calculate PCK from the predicted heatmaps.
        acc = accuracy(heatmaps, target.cpu(), acc_joints)

        # Calculate locations in original image space from the predicted heatmaps.
        preds = final_preds(heatmaps, meta['center'], meta['scale'], [64, 64])
        # NEW for visualization: (and redundant, but for visualization)
        preds_unprocessed, preds_unprocessed_maxval = get_preds_soft(heatmaps, return_maxval=True)
        # preds_unprocessed, preds_unprocessed_norm, preds_unprocessed_maxval = get_preds_soft(heatmaps, return_maxval=True, norm_and_unnorm_coords=True)

        ind = 0
        for example_index, pose in zip(meta['index'], preds):
            predictions[example_index] = pose
            # NEW for visualization
            if save_imgs_path is not None:
                out_name = os.path.join(save_imgs_path, 'res_' + str( example_index.item()) + '.png')
                pred_unp = preds_unprocessed[ind, :, :]

                pred_unp_maxval = preds_unprocessed_maxval[ind, :, :]
                pred_unp_prep = torch.cat((pred_unp, pred_unp_maxval), 1)
                inp_img = input[ind, :, :, :]
                # the following line (with -1) should not be needed anymore after cvpr (after bugfix01 in data preparation 08.09.2022)
                # pred_unp_prep[:, :2] = pred_unp_prep[:, :2] - 1
                # save_input_image_with_keypoints(inp_img, pred_unp_prep, out_path=out_name, threshold=0.1, print_scores=True)    # here we have default ratio_in_out=4.
                # NEW: 08.09.2022 after bugfix01
                pred_unp_prep[:, :2] = pred_unp_prep[:, :2] * 4 

                if 'name' in meta.keys():       # we do this for the stanext set
                    name = meta['name'][ind]
                    out_path_keyp_img = os.path.join(os.path.dirname(out_name), name)
                    out_path_json = os.path.join(os.path.dirname(out_name), name).replace('_vis', '_json').replace('.jpg', '.json')
                    if not os.path.exists(os.path.dirname(out_path_json)):
                        os.makedirs(os.path.dirname(out_path_json))
                    if not os.path.exists(os.path.dirname(out_path_keyp_img)):
                        os.makedirs(os.path.dirname(out_path_keyp_img))  
                    save_input_image_with_keypoints(inp_img, pred_unp_prep, out_path=out_path_keyp_img, ratio_in_out=1.0, threshold=0.1, print_scores=True)    # threshold=0.3
                    out_name_json = out_path_json   # os.path.join(save_imgs_path, 'res_' + str( example_index.item()) + '.json')
                    res_dict = {
                        'pred_joints_256': list(pred_unp_prep.cpu().numpy().astype(float).reshape((-1))),
                        'center': list(meta['center'][ind, :].cpu().numpy().astype(float).reshape((-1))),
                        'scale': meta['scale'][ind].item()}
                    with open(out_name_json, 'w') as outfile: json.dump(res_dict, outfile)
                else:
                    save_input_image_with_keypoints(inp_img, pred_unp_prep, out_path=out_name, ratio_in_out=1.0, threshold=0.1, print_scores=True)    # threshold=0.3

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
