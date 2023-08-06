

import os
import glob
import numpy as np
import math
import torch
import torch.utils.data as data

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.anipose_data_info import COMPLETE_DATA_INFO        
from stacked_hourglass.utils.imutils import load_image, im_to_torch 
from stacked_hourglass.utils.transforms import crop, color_normalize
from stacked_hourglass.utils.pilutil import imresize 
from stacked_hourglass.utils.imutils import im_to_torch
from configs.data_info import COMPLETE_DATA_INFO_24


class ImgCrops(data.Dataset):
    DATA_INFO = COMPLETE_DATA_INFO_24
    ACC_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16]  

    def __init__(self, image_list, bbox_list=None, inp_res=256, dataset_mode='keyp_only'):
        # the list contains the images directly, not only their paths
        self.image_list = image_list
        self.bbox_list = bbox_list
        self.inp_res = inp_res
        self.test_name_list = []
        for ind in np.arange(0, len(self.image_list)):
            self.test_name_list.append(str(ind))
        print('len(dataset): ' + str(self.__len__()))

    def __getitem__(self, index):

        # load image
        img = im_to_torch(self.image_list[index])

        # try loading bounding box
        if self.bbox_list is not None:
            bbox = self.bbox_list[index]
            bbox_xywh = [bbox[0][0], bbox[0][1], bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1]]
            bbox_c = [bbox_xywh[0]+0.5*bbox_xywh[2], bbox_xywh[1]+0.5*bbox_xywh[3]]
            bbox_max = max(bbox_xywh[2], bbox_xywh[3])
            bbox_diag = math.sqrt(bbox_xywh[2]**2 + bbox_xywh[3]**2)
            bbox_s = bbox_max / 200. * 256. / 200.  # maximum side of the bbox will be 200
            c = torch.Tensor(bbox_c)
            s = bbox_s
            img_prep = crop(img, c, s, [self.inp_res, self.inp_res], rot=0)
        else:
            # prepare image (cropping and color)
            img_max = max(img.shape[1], img.shape[2])
            img_padded = torch.zeros((img.shape[0], img_max, img_max))
            if img_max == img.shape[2]:
                start = (img_max-img.shape[1])//2
                img_padded[:, start:start+img.shape[1], :] = img
            else:
                start = (img_max-img.shape[2])//2
                img_padded[:, :, start:start+img.shape[2]] = img   
            img = img_padded
            img_prep = im_to_torch(imresize(img, [self.inp_res, self.inp_res], interp='bilinear'))   
        
        inp = color_normalize(img_prep, self.DATA_INFO.rgb_mean, self.DATA_INFO.rgb_stddev)
        # add the following fields to make it compatible with stanext, most of them are fake
        target_dict = {'index': index, 'center' : -2, 'scale' : -2, 
            'breed_index': -2, 'sim_breed_index': -2,
            'ind_dataset': 1}
        target_dict['pts'] = np.zeros((self.DATA_INFO.n_keyp, 3))
        target_dict['tpts'] = np.zeros((self.DATA_INFO.n_keyp, 3))
        target_dict['target_weight'] = np.zeros((self.DATA_INFO.n_keyp, 1))
        target_dict['silh'] = np.zeros((self.inp_res, self.inp_res))
        return inp, target_dict


    def __len__(self):
        return len(self.image_list)   









