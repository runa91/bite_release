import gzip
import json
import os
import glob
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
import xml.etree.ElementTree as ET

from csv import DictReader
from pycocotools.mask import decode as decode_RLE

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from src.configs.anipose_data_info import COMPLETE_DATA_INFO
from src.stacked_hourglass.utils.imutils import load_image, draw_labelmap, draw_multiple_labelmaps
from src.stacked_hourglass.utils.misc import to_torch
from src.stacked_hourglass.utils.transforms import shufflelr, crop, color_normalize, fliplr, transform
import src.stacked_hourglass.datasets.utils_stanext as utils_stanext 
from src.stacked_hourglass.utils.visualization import save_input_image_with_keypoints


class AniPose(data.Dataset):
    DATA_INFO = COMPLETE_DATA_INFO

    # Suggested joints to use for average PCK calculations.
    ACC_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]      # don't know ...

    def __init__(self, image_path=None, is_train=True, inp_res=256, out_res=64, sigma=1,
                 scale_factor=0.25, rot_factor=30, label_type='Gaussian', 
                 do_augment='default', shorten_dataset_to=None, dataset_mode='keyp_only'):
        # self.img_folder_mpii = image_path # root image folders
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
        if self.dataset_mode=='complete' or self.dataset_mode=='keyp_and_seg':
            self.calc_seg = True
        else:
            self.calc_seg = False

        self.kp_dict = self.keyp_name_to_ind()

        # import pdb; pdb.set_trace()

        self.top_folder = '...../Animals/data/animal_pose_dataset/'
        self.folder_imgs_0 = '/ps/project/datasets/VOCdevkit/VOC2012/JPEGImages/'
        self.folder_imgs_1 = os.path.join(self.top_folder, 'animalpose_image_part2', 'dog')
        self.folder_annot_0 = os.path.join(self.top_folder, 'PASCAL2011_animal_annotation', 'dog')
        self.folder_annot_1 = os.path.join(self.top_folder, 'animalpose_anno2', 'dog')
        all_annot_files_0 = glob.glob(self.folder_annot_0 + '/*.xml')       # 1571
        '''all_annot_files_0_raw.sort()
        all_annot_files_0 = []                                                  # 1331
        for ind_f, f in enumerate(all_annot_files_0_raw):
            name = (f.split('/')[-1]).split('.xml')[0]
            name_main = name[:-2]
            if ind_f > 0:
                if (not name_main == name_main_last) or (ind_f == len(all_annot_files_0_raw)-1):
                    all_annot_files_0.append(f_last)
            f_last = f
            name_main_last = name_main'''
        all_annot_files_1 = glob.glob(self.folder_annot_1 + '/*.xml')       #  200
        all_annot_files = all_annot_files_0 + all_annot_files_1

        # new for hg_anipose_v1
        self.train_name_list = all_annot_files[:-50]
        self.test_name_list = all_annot_files[-50:] 

        print('anipose dataset size: ')
        print(len(self.train_name_list))
        print(len(self.test_name_list))


    # ----------------------------------------- 
    def read_content(sewlf, xml_file, annot_type='animal_pose'):
        # annot_type is either 'animal_pose' or 'animal_pose_voc' or 'voc'
        # examples:
        #   animal_pose: '..../Animals/data/animal_pose_dataset/animalpose_anno2/cat/ca137.xml'
        #   animal_pose_voc: '..../Animals/data/animal_pose_dataset/PASCAL2011_animal_annotation/cat/2008_005380_1.xml'
        #   voc: '.../VOCdevkit/VOC2012/Annotations/2011_000192.xml'
        if annot_type == 'animal_pose' or annot_type == 'animal_pose_voc':
            my_dict = {}
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for child in root:  # list
                if child.tag == 'image':
                    my_dict['image'] = child.text
                elif child.tag == 'category':
                    my_dict['category'] = child.text
                elif child.tag == 'visible_bounds':
                    my_dict['visible_bounds'] = child.attrib
                elif child.tag == 'keypoints':
                    n_kp = len(child)
                    xyzvis = np.zeros((n_kp, 4))
                    kp_names = []
                    for ind_kp, kp in enumerate(child):    # list
                        xyzvis[ind_kp, 0] = kp.attrib['x']
                        xyzvis[ind_kp, 1] = kp.attrib['y']
                        xyzvis[ind_kp, 2] = kp.attrib['z']
                        xyzvis[ind_kp, 3] = kp.attrib['visible']
                        kp_names.append(kp.attrib['name'])
                    my_dict['keypoints_xyzvis'] = xyzvis
                    my_dict['keypoints_names'] = kp_names
                elif child.tag == 'voc_id':             # animal_pose_voc only
                    my_dict['voc_id'] = child.text
                elif child.tag == 'polylinesegments':   # animal_pose_voc only
                    my_dict['polylinesegments'] = child[0].attrib
                else:
                    print('tag does not exist: ' + child.tag)
            # print(my_dict)
        elif annot_type == 'voc':
            my_dict = {}
            print('not yet read')
        else:
            print('this annot_type does not exist')
            import pdb; pdb.set_trace()
        return my_dict


    def keyp_name_to_ind(self):
        '''AniPose_JOINT_NAMES = [
            'L_Eye', 'R_Eye', 'Nose', 'L_EarBase', 'Throat', 'R_F_Elbow', 'R_F_Paw', 
            'R_B_Paw', 'R_EarBase', 'L_F_Elbow', 'L_F_Paw', 'Withers', 'TailBase', 
            'L_B_Paw', 'L_B_Elbow', 'R_B_Elbow', 'L_F_Knee', 'R_F_Knee', 'L_B_Knee', 
            'R_B_Knee']'''
        kps = self.DATA_INFO.joint_names
        kps_dict = {}
        for ind_kp, kp in enumerate(kps):
            kps_dict[kp] = ind_kp
            kps_dict[kp.lower()] = ind_kp
            if kp.lower() == 'l_earbase':
                kps_dict['l_ear'] = ind_kp
            if kp.lower() == 'r_earbase':
                kps_dict['r_ear'] = ind_kp
            if kp.lower() == 'tailbase':
                kps_dict['tail'] = ind_kp
        return kps_dict



    def __getitem__(self, index):

        if self.is_train:
            xml_path = self.train_name_list[index]
        else:
            xml_path = self.test_name_list[index]

        name = (xml_path.split('/')[-1]).split('.xml')[0]
        annot_dict = self.read_content(xml_path, annot_type='animal_pose_voc')

        if xml_path.split('/')[-3] == 'PASCAL2011_animal_annotation':
            img_path = os.path.join(self.folder_imgs_0, annot_dict['image'] + '.jpg')
            keyword_ymin = 'ymin'
        else:
            img_path = os.path.join(self.folder_imgs_1, annot_dict['image'])
            keyword_ymin = 'xmax'

        sf = self.scale_factor
        rf = self.rot_factor

        vis_np = np.zeros((self.DATA_INFO.n_keyp))
        pts_np = np.ones((self.DATA_INFO.n_keyp, 2)) * (-1000)
        for ind_key, key in enumerate(annot_dict['keypoints_names']):
            key_lower = key.lower()
            ind_new = self.kp_dict[key_lower]
            vis_np[ind_new] = annot_dict['keypoints_xyzvis'][ind_key, 3]
            # remark: the first training run (animalpose_hg8_v0) was without subtracting 1 which would be important!
            pts_np[ind_new] = annot_dict['keypoints_xyzvis'][ind_key, 0:2] - 1

        pts_np = np.concatenate((pts_np, vis_np[:, None]), axis=1)
        pts = torch.Tensor(pts_np)

        # what we were doing until 08.09.2022:
        # bbox_xywh = [float(annot_dict['visible_bounds']['xmin']), float(annot_dict['visible_bounds'][keyword_ymin]), \
        #             float(annot_dict['visible_bounds']['width']), float(annot_dict['visible_bounds']['height'])]
        bbox_xywh = [float(annot_dict['visible_bounds']['xmin'])-1, float(annot_dict['visible_bounds'][keyword_ymin])-1, \
                    float(annot_dict['visible_bounds']['width']), float(annot_dict['visible_bounds']['height'])]

        bbox_c = [bbox_xywh[0]+0.5*bbox_xywh[2], bbox_xywh[1]+0.5*bbox_xywh[3]]
        bbox_max = max(bbox_xywh[2], bbox_xywh[3])
        bbox_diag = math.sqrt(bbox_xywh[2]**2 + bbox_xywh[3]**2)
        # bbox_s = bbox_max / 200.      # the dog will fill the image -> bbox_max = 256
        # bbox_s = bbox_diag / 200.     # diagonal of the boundingbox will be 200
        bbox_s = bbox_max / 200. * 256. / 200.  # maximum side of the bbox will be 200
        c = torch.Tensor(bbox_c)
        s = bbox_s

        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.size(0)
        img = load_image(img_path)  # CxHxW

        # segmentation map (we reshape it to 3xHxW, such that we can do the 
        #   same transformations as with the image)
        if self.calc_seg:
            raise NotImplementedError
            seg = torch.Tensor(utils_stanext.get_seg_from_entry(data)[None, :, :])
            seg = torch.cat(3*[seg])

        r = 0
        # self.is_train = False
        do_flip = False
        if self.do_augment:
            s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0
            # Flip
            if random.random() <= 0.5:
                do_flip = True
                img = fliplr(img)
                if self.calc_seg:
                    seg = fliplr(seg)
                # pts = shufflelr(pts, img.size(2), self.DATA_INFO.hflip_indices)
                # remark: for BITE we figure out that a -1 was missing in the point mirroring term
                # idea:
                #   image coordinates are 0, 1, 2, 3
                #   image size is 4
                #   the new point location for former 0 should be 3 and not 4!
                pts = shufflelr(pts, img.size(2)-1, self.DATA_INFO.hflip_indices)
                c[0] = img.size(2) - c[0] - 1
            # Color
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Prepare image and groundtruth map
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = color_normalize(inp, self.DATA_INFO.rgb_mean, self.DATA_INFO.rgb_stddev)
        if self.calc_seg:
            seg = crop(seg, c, s, [self.inp_res, self.inp_res], rot=r)

        # Generate ground truth
        tpts = pts.clone()
        target_weight = tpts[:, 2].clone().view(nparts, 1)

        target = torch.zeros(nparts, self.out_res, self.out_res)
        for i in range(nparts):
            # if tpts[i, 2] > 0: # This is evil!!
            '''if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2], c, s, [self.out_res, self.out_res], rot=r, as_int=False))
                target[i], vis = draw_labelmap(target[i], tpts[i], self.sigma, type=self.label_type)
                target_weight[i, 0] *= vis'''
            if tpts[i, 1] > 0:
                # this pytorch function (transforms) assumes that coordinates which start at 1 instead of 0!
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2]+1, c, s, [self.out_res, self.out_res], rot=r, as_int=False)) - 1
                target[i], vis = draw_labelmap(target[i], tpts[i], self.sigma, type=self.label_type)
                target_weight[i, 0] *= vis

        meta = {'index' : index, 'center' : c, 'scale' : s,
                'pts' : pts, 'tpts' : tpts, 'target_weight': target_weight}

        if self.dataset_mode=='keyp_only':
            return inp, target, meta
        elif self.dataset_mode=='keyp_and_seg':
            raise NotImplementedError
            meta['silh'] = seg[0, :, :]
            meta['name'] = name
            return inp, target, meta
        elif self.dataset_mode=='complete':
            raise NotImplementedError
            target_dict = meta
            target_dict['silh'] = seg[0, :, :]
            # NEW for silhouette loss
            distmat_tofg = ndimage.distance_transform_edt(1-target_dict['silh'])    # values between 0 and up to 100 or more
            target_dict['silh_distmat_tofg'] = distmat_tofg     
            distmat_tobg = ndimage.distance_transform_edt(target_dict['silh'])    
            target_dict['silh_distmat_tobg'] = distmat_tobg    
            return inp, target_dict
        else:
            raise ValueError



    def __len__(self):
        if self.is_train:
            return len(self.train_name_list)  # len(self.train_list)
        else:
            return len(self.test_name_list)   # len(self.valid_list)


