# 24 joints instead of 20!!


import gzip
import json
import os
import random
import math
import numpy as np
import torch
import torch.utils.data as data
from importlib_resources import open_binary
from scipy.io import loadmat
from tabulate import tabulate
import itertools
import json
from scipy import ndimage

from csv import DictReader
from pycocotools.mask import decode as decode_RLE

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.configs.data_info import COMPLETE_DATA_INFO_24
from src.stacked_hourglass.utils.imutils import load_image, draw_labelmap, draw_multiple_labelmaps
from src.stacked_hourglass.utils.misc import to_torch
from src.stacked_hourglass.utils.transforms import shufflelr, crop, color_normalize, fliplr, transform
import src.stacked_hourglass.datasets.utils_stanext as utils_stanext 
from src.stacked_hourglass.utils.visualization import save_input_image_with_keypoints



class DogsVOC(data.Dataset):
    DATA_INFO = COMPLETE_DATA_INFO_24

    # Suggested joints to use for average PCK calculations.
    ACC_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16]      # don't know ...

    def __init__(self, image_path=None, is_train=True, inp_res=256, out_res=64, sigma=1,
                 scale_factor=0.25, rot_factor=30, label_type='Gaussian', 
                 do_augment='default', shorten_dataset_to=None, dataset_mode='keyp_only', V12=None):
        # self.img_folder_mpii = image_path # root image folders
        self.V12 = V12
        self.is_train = is_train # training set or test set
        if do_augment == 'yes':
            self.do_augment = True
        elif do_augment == 'no':
            self.do_augment = False
        elif do_augment=='default':
            if self.is_train:
                self.do_augment = True
            else:
                self.do_augment = False
        else:
            raise ValueError
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type
        self.dataset_mode = dataset_mode
        if self.dataset_mode=='complete' or self.dataset_mode=='keyp_and_seg' or self.dataset_mode=='keyp_and_seg_and_partseg':
            self.calc_seg = True
        else:
            self.calc_seg = False

        # create train/val split
        # REMARK: I assume we should have a different train / test split here
        self.img_folder = utils_stanext.get_img_dir(V12=self.V12)
        self.train_dict, self.test_dict, self.val_dict = utils_stanext.load_stanext_json_as_dict(split_train_test=True, V12=self.V12)
        self.train_name_list = list(self.train_dict.keys())     # 7004
        self.test_name_list = list(self.test_dict.keys())       # 5031

        # breed json_path
        breed_json_path = '...../Animals/data/dog_datasets/Stanford_Dogs_Dataset/StanfordExtra/StanExt_breed_dict_v2.json'

        # only use images that show fully visible dogs in standing or walking poses
        self.train_name_list = sorted(self.train_name_list)
        self.test_name_list = sorted(self.test_name_list)

        random.seed(4)
        random.shuffle(self.train_name_list)
        random.shuffle(self.test_name_list)


        if shorten_dataset_to is not None:
            self.train_name_list = self.train_name_list[0 : min(len(self.train_name_list), shorten_dataset_to)]
            self.test_name_list = self.test_name_list[0 : min(len(self.test_name_list), shorten_dataset_to)]

            if shorten_dataset_to == 12:
                # my_sample = self.test_name_list[2]        # black haired dog
                my_sample = self.test_name_list[2]
                for ind in range(0, 12):
                    self.test_name_list[ind] = my_sample

        # add results for eyes, whithers and throat as obtained through anipose
        self.path_anipose_out_root = '...../Animals/data/dog_datasets/Stanford_Dogs_Dataset/StanfordExtra/animalpose_hg8_v0_results_on_StanExt/'

        self.dogvoc_path_root = '...../Animals/data/pascal_voc_parts/'
        self.dogvoc_path_images = self.dogvoc_path_root + 'dog_images/' 
        self.dogvoc_path_masks = self.dogvoc_path_root + 'dog_masks/'

        with open(self.dogvoc_path_masks + 'voc_dogs_bodypart_info.json', 'r') as file:
            self.body_part_info = json.load(file)
        with open(self.dogvoc_path_masks + 'voc_dogs_train.json', 'r') as file:
            train_set_init = json.load(file)   # 707
        with open(self.dogvoc_path_masks + 'voc_dogs_val.json', 'r') as file:
            val_set_init = json.load(file)     # 709
        self.train_set = train_set_init + val_set_init[:-36]
        self.val_set = val_set_init[-36:]

        print('len(dataset): ' + str(self.__len__()))
        # print(self.test_name_list[0:10])

    def get_body_part_indices(self):
        silh = [
            ('background', [0]),
            ('foreground', [255, 21, 57, 30, 59, 34, 48, 50, 79, 49, 61, 60, 54, 53, 36, 35, 27, 26, 78])]
        full_body = [
            ('other', [255]),
            ('head', [21, 57, 30, 59, 34, 48, 50]),
            ('torso', [79, 49]),
            ('right front leg', [61, 60]),
            ('right back leg', [54, 53]),
            ('left front leg', [36, 35]),
            ('left back leg', [27, 26]),
            ('tail', [78])]
        head = [
            ('other', [21, 59, 34]),
            ('right ear', [57]),
            ('left ear', [30]),
            ('muzzle', [48]),
            ('nose', [50])]
        torso = [
            ('other', [79]),    # wrong 34
            ('neck', [49])]
        all_parts = {
            'silh': silh,
            'full_body': full_body,
            'head': head,
            'torso': torso}
        return all_parts





    def __getitem__(self, index):

        if self.is_train:
            name = self.train_name_list[index]
            data = self.train_dict[name]
            # data = utils_stanext.get_dog(self.train_dict, name)
        else:
            name = self.test_name_list[index]
            data = self.test_dict[name]
            # data = utils_stanext.get_dog(self.test_dict, name)

        if self.is_train:
            img_info = self.train_set[index]
        else:
            img_info = self.val_set[index]

        sf = self.scale_factor
        rf = self.rot_factor

        img_path = os.path.join(self.dogvoc_path_images, img_info['img_name'])

        # bbox_yxhw = img_info['bbox']
        # bbox_xywh = [bbox_yxhw[1], bbox_yxhw[0], bbox_yxhw[2], bbox_yxhw[3]]
        bbox_xywh = img_info['bbox']
        bbox_c = [bbox_xywh[0]+0.5*bbox_xywh[2], bbox_xywh[1]+0.5*bbox_xywh[3]]
        bbox_max = max(bbox_xywh[2], bbox_xywh[3])
        bbox_diag = math.sqrt(bbox_xywh[2]**2 + bbox_xywh[3]**2)
        # bbox_s = bbox_max / 200.      # the dog will fill the image -> bbox_max = 256
        # bbox_s = bbox_diag / 200.     # diagonal of the boundingbox will be 200
        bbox_s = bbox_max / 200. * 256. / 200.  # maximum side of the bbox will be 200
        c = torch.Tensor(bbox_c)
        s = bbox_s

        # For single-person pose estimation with a centered/scaled figure
        img = load_image(img_path)  # CxHxW

        # segmentation map (we reshape it to 3xHxW, such that we can do the 
        #   same transformations as with the image)
        if self.do_augment and (random.random() <= 0.5):
            do_flip = True
        else:
            do_flip = False

        if self.calc_seg:
            mask = np.load(os.path.join(self.dogvoc_path_masks, img_info['img_name'].split('.')[0] + '_' + str(img_info['ind_bbox']) + '.npz.npy'))    
            seg_np = mask.copy()
            seg_np[mask==0] = 0
            seg_np[mask>0] = 1
            seg = torch.Tensor(seg_np[None, :, :])
            seg = torch.cat(3*[seg])

            # NEW: body parts
            all_parts = self.get_body_part_indices()
            body_part_index_list = []
            body_part_name_list = []
            n_tbp = 3
            n_bp = 15
            # body_part_matrix_multiple_hot = np.zeros((n_bp, mask.shape[0], mask.shape[1]))
            body_part_matrix_np = np.ones((n_tbp, mask.shape[0], mask.shape[1])) * (-1)
            ind_bp = 0
            for ind_tbp, part in enumerate(['full_body', 'head', 'torso']):
                # import pdb; pdb.set_trace()
                if part == 'full_body':
                    inds_mirr = [0, 1, 2, 5, 6, 3, 4, 7]
                elif part == 'head':
                    inds_mirr = [0, 2, 1, 3, 4]
                else:
                    inds_mirr = [0, 1]
                for ind_sbp, subpart in enumerate(all_parts[part]):
                    if do_flip:
                        ind_sbp_corr = inds_mirr[ind_sbp]      # we use this if the image is mirrored later on
                    else:
                        ind_sbp_corr = ind_sbp
                    bp_name = subpart[0]
                    bp_indices = subpart[1]
                    body_part_index_list.append(bp_indices)
                    body_part_name_list.append(bp_name)
                    # create matrix slice
                    xx = [mask==ind for ind in bp_indices]
                    xx_mat = (np.stack(xx).sum(axis=0))
                    # body_part_matrix_multiple_hot[ind_bp, :, :] = xx_mat
                    # add to matrix
                    body_part_matrix_np[ind_tbp, xx_mat>0] = ind_sbp_corr
                    ind_bp += 1
            body_part_weight_masks_np = np.zeros((n_tbp, mask.shape[0], mask.shape[1]))
            body_part_weight_masks_np[0, mask>0] = 1   # full body
            body_part_weight_masks_np[1, body_part_matrix_np[0, :, :]==1] = 1   # head
            body_part_weight_masks_np[2, body_part_matrix_np[0, :, :]==2] = 1   # torso
            body_part_matrix_np[body_part_weight_masks_np==0] = 16
            body_part_matrix = torch.Tensor(body_part_matrix_np + 2.0)  # / 100

            # import pdb; pdb.set_trace()

            bbox_c_int0 = [int(bbox_c[0]), int(bbox_c[1])]
            bbox_c_int1 = [int(bbox_c[0])+10, int(bbox_c[1])+10]
            '''bpm_c0 = body_part_matrix[:, bbox_c_int0[1], bbox_c_int0[0]].clone()
            bpm_c1 = body_part_matrix[:, bbox_c_int1[1], bbox_c_int1[0]].clone()
            zero_replacement = torch.Tensor([0, 0, 0.99])
            body_part_matrix[:, bbox_c_int0[1], bbox_c_int0[0]] = zero_replacement
            body_part_matrix[:, bbox_c_int1[1], bbox_c_int1[0]] = 1'''
            ii = 3
            bpm_c0 = body_part_matrix[2, bbox_c_int0[1]-ii:bbox_c_int0[1]+ii, bbox_c_int0[0]-ii:bbox_c_int0[0]+ii]
            bpm_c1 = body_part_matrix[2, bbox_c_int1[1]-ii:bbox_c_int1[1]+ii, bbox_c_int1[0]-ii:bbox_c_int1[0]+ii]
            body_part_matrix[2, bbox_c_int0[1]-ii:bbox_c_int0[1]+ii, bbox_c_int0[0]-ii:bbox_c_int0[0]+ii] = 0
            body_part_matrix[2, bbox_c_int1[1]-ii:bbox_c_int1[1]+ii, bbox_c_int1[0]-ii:bbox_c_int1[0]+ii] = 255
            body_part_matrix = (body_part_matrix).long()
            # body_part_name_list
            # ['other', 'head', 'torso', 'right front leg', 'right back leg', 'left front leg', 'left back leg', 'tail', 'other', 'right ear', 'left ear', 'muzzle', 'nose', 'other', 'neck']
            # swap indices:
            # bp_mirroring_inds = [0, 1, 2, 5, 6, 3, 4, 7, 8, 10, 9, 11, 12, 13, 14]


        r = 0
        # self.is_train = False
        if self.do_augment:
            s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0
            # Flip
            if do_flip:
                img = fliplr(img)
                if self.calc_seg:
                    seg = fliplr(seg)
                    body_part_matrix = fliplr(body_part_matrix)
                c[0] = img.size(2) - c[0]
            # Color
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Prepare image and groundtruth map
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = color_normalize(inp, self.DATA_INFO.rgb_mean, self.DATA_INFO.rgb_stddev)

        # import pdb; pdb.set_trace()

        if self.calc_seg:
            seg = crop(seg, c, s, [self.inp_res, self.inp_res], rot=r)

            # 'crop' will divide by 255 and perform zero padding (
            #   -> weird function that tries to rescale! Because of that I add zeros and ones in the beginning
            xx = body_part_matrix.clone()

            body_part_matrix = crop(body_part_matrix, c, s, [self.inp_res, self.inp_res], rot=r, interp='nearest')  
                    
            body_part_matrix = body_part_matrix*255 - 2

            body_part_matrix[body_part_matrix == -2] = -1
            body_part_matrix[body_part_matrix == 16] = -1
            body_part_matrix[body_part_matrix == 253] = -1


        # Generate ground truth
        nparts = 24
        target_weight = torch.zeros(nparts, 1)
        target = torch.zeros(nparts, self.out_res, self.out_res)
        pts = torch.zeros((nparts, 3))                        
        tpts = torch.zeros((nparts, 3))                        

        # import pdb; pdb.set_trace()


        # meta = {'index' : index, 'center' : c, 'scale' : s, 'do_flip' : do_flip, 'rot' : r, 'resolution' : [self.out_res, self.out_res], 'name' : name,
        #     'pts' : pts, 'tpts' : tpts, 'target_weight': target_weight, 'breed_index': this_breed['index']}
        # meta = {'index' : index, 'center' : c, 'scale' : s, 'do_flip' : do_flip, 'rot' : r, 'resolution' : self.out_res,
        #     'pts' : pts, 'tpts' : tpts, 'target_weight': target_weight, 'breed_index': this_breed['index']}   
        # meta = {'index' : index, 'center' : c, 'scale' : s,
        #     'pts' : pts, 'tpts' : tpts, 'target_weight': target_weight, 
        #    'breed_index': this_breed['index'], 'sim_breed_index': sim_breed_index,
        #    'ind_dataset': 0}   # ind_dataset: 0 for stanext or stanexteasy or stanext 24
        meta = {'index' : index, 'center' : c, 'scale' : s,
            'pts' : pts, 'tpts' : tpts, 'target_weight': target_weight, 
           'ind_dataset': 3} 

        #import pdb; pdb.set_trace()


        if self.dataset_mode=='keyp_and_seg_and_partseg':
            # meta = {}
            meta['silh'] = seg[0, :, :]
            meta['name'] = name
            meta['body_part_matrix'] = body_part_matrix.long()
            # meta['body_part_weights'] = body_part_weight_masks 
            # import pdb; pdb.set_trace()
            return inp, target, meta
        else:
            raise ValueError



    def __len__(self):
        if self.is_train:
            return len(self.train_set)  
        else:
            return len(self.val_set)


