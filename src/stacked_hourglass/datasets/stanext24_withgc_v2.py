# this version includes all ground contact labeled data, not only the sitting/lying poses


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
import csv
import pickle as pkl

from csv import DictReader
from pycocotools.mask import decode as decode_RLE

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.data_info import COMPLETE_DATA_INFO_24
from stacked_hourglass.utils.imutils import load_image, draw_labelmap, draw_multiple_labelmaps
from stacked_hourglass.utils.misc import to_torch
from stacked_hourglass.utils.transforms import shufflelr, crop, color_normalize, fliplr, transform
import stacked_hourglass.datasets.utils_stanext as utils_stanext 
from stacked_hourglass.utils.visualization import save_input_image_with_keypoints
from configs.dog_breeds.dog_breed_class import COMPLETE_ABBREV_DICT, COMPLETE_SUMMARY_BREEDS, SIM_MATRIX_RAW, SIM_ABBREV_INDICES
from configs.dataset_path_configs import STANEXT_RELATED_DATA_ROOT_DIR
from smal_pytorch.smal_model.smal_basics import get_symmetry_indices


def read_csv(csv_file):
    with open(csv_file,'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        row_list = [{h:x for (h,x) in zip(headers,row)} for row in reader]
    return row_list

class StanExtGC(data.Dataset):
    DATA_INFO = COMPLETE_DATA_INFO_24

    # Suggested joints to use for keypoint reprojection error calculations 
    ACC_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16]      

    def __init__(self, image_path=None, is_train=True, inp_res=256, out_res=64, sigma=1,
                 scale_factor=0.25, rot_factor=30, label_type='Gaussian', 
                 do_augment='default', shorten_dataset_to=None, dataset_mode='keyp_only', V12=None, val_opt='test', add_nonflat=False):
        self.V12 = V12
        self.is_train = is_train    # training set or test set
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
        self.add_nonflat = add_nonflat
        if self.dataset_mode=='complete' or self.dataset_mode=='complete_with_gc' or self.dataset_mode=='keyp_and_seg' or self.dataset_mode=='keyp_and_seg_and_partseg':
            self.calc_seg = True
        else:
            self.calc_seg = False
        self.val_opt = val_opt

        # create train/val split
        self.img_folder = utils_stanext.get_img_dir(V12=self.V12)
        self.train_dict, init_test_dict, init_val_dict = utils_stanext.load_stanext_json_as_dict(split_train_test=True, V12=self.V12)
        self.train_name_list = list(self.train_dict.keys())     # 7004
        if self.val_opt == 'test':
            self.test_dict = init_test_dict
            self.test_name_list = list(self.test_dict.keys())        
        elif self.val_opt == 'val':
            self.test_dict = init_val_dict
            self.test_name_list = list(self.test_dict.keys())       
        else:
            raise NotImplementedError


        # import pdb; pdb.set_trace()


        # path_gc_annots_overview = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stage3/gc_annots_overview_first699.pkl'
        path_gc_annots_overview_stage3 = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stage3/gc_annots_overview_stage3complete.pkl'
        with open(path_gc_annots_overview_stage3, 'rb') as f:
            self.gc_annots_overview_stage3 = pkl.load(f)                        # 2346

        path_gc_annots_overview_stage2b_contact = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stage2b/gc_annots_overview_stage2b_contact_complete.pkl'
        with open(path_gc_annots_overview_stage2b_contact, 'rb') as f:
            self.gc_annots_overview_stage2b_contact = pkl.load(f)               # 832
        
        path_gc_annots_overview_stage2b_nocontact = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stage2b/gc_annots_overview_stage2b_nocontact_complete.pkl'
        with open(path_gc_annots_overview_stage2b_nocontact, 'rb') as f:
            self.gc_annots_overview_stage2b_nocontact = pkl.load(f)             # 32

        path_gc_annots_overview_stages12_all4pawsincontact = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stages12together/gc_annots_overview_all4pawsincontact.pkl'
        with open(path_gc_annots_overview_stages12_all4pawsincontact, 'rb') as f:
            self.gc_annots_overview_stages12_all4pawsincontact = pkl.load(f)    # 1, symbolic only

        path_gc_annots_categories_stages12 = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stages12together/gc_annots_categories_stages12_complete.pkl'
        with open(path_gc_annots_categories_stages12, 'rb') as f:
            self.gc_annots_categories = pkl.load(f)                             # 12538


        test_name_list_gc = []
        for name in self.test_name_list:
            if name in self.gc_annots_categories.keys():
                value = self.gc_annots_categories[name]
                if (value['is_vis'] in [True, None]) and (value['is_flat'] in [True, None]) and (not value['pose'] == 'cantsee'):
                    test_name_list_gc.append(name)

        train_name_list_gc = []
        for name in self.train_name_list:
                value = self.gc_annots_categories[name]
                if (value['is_vis'] in [True, None]) and (value['is_flat'] in [True, None]) and (not value['pose'] == 'cantsee'):
                    train_name_list_gc.append(name)

        # import pdb; pdb.set_trace()





        '''self.gc_annots_overview = self.gc_annots_overview_stage3
        list_gc_labelled_images = list(self.gc_annots_overview.keys())

        test_name_list_gc = []
        for name in self.test_name_list:
            if name.split('.')[0] in list_gc_labelled_images:
                test_name_list_gc.append(name)

        train_name_list_gc = []
        for name in self.train_name_list:
            if name.split('.')[0] in list_gc_labelled_images:
                train_name_list_gc.append(name)'''

        random.seed(4)
        random.shuffle(test_name_list_gc)




        # new: add images with non-flat ground in the end
        # import pdb; pdb.set_trace()
        if self.add_nonflat:
            self.train_name_list_nonflat = []
            for name in self.train_name_list:
                if name in self.gc_annots_categories.keys():
                    value = self.gc_annots_categories[name]
                    if (value['is_vis'] in [True, None]) and (value['is_flat'] in [False]):
                        self.train_name_list_nonflat.append(name)
            self.test_name_list_nonflat = []
            for name in self.test_name_list:
                if name in self.gc_annots_categories.keys():
                    value = self.gc_annots_categories[name]
                    if (value['is_vis'] in [True, None]) and (value['is_flat'] in [False]):
                        self.test_name_list_nonflat.append(name)




        self.test_name_list = test_name_list_gc
        self.train_name_list = train_name_list_gc





        '''
        already_labelled  = ['n02093991-Irish_terrier/n02093991_2874.jpg',
                            'n02093754-Border_terrier/n02093754_1062.jpg',
                            'n02092339-Weimaraner/n02092339_1672.jpg',
                            'n02096177-cairn/n02096177_4916.jpg',
                            'n02110185-Siberian_husky/n02110185_725.jpg',
                            'n02110806-basenji/n02110806_761.jpg',
                            'n02094433-Yorkshire_terrier/n02094433_2474.jpg',
                            'n02097474-Tibetan_terrier/n02097474_8796.jpg',
                            'n02099601-golden_retriever/n02099601_2495.jpg']
        self.trainvaltest_dict = dict(self.train_dict)
        for d in (init_test_dict, init_val_dict): self.trainvaltest_dict.update(d)

        gc_annot_csv = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/my_gcannotations_qualification.csv'
        gc_row_list = read_csv(gc_annot_csv)

        json_acceptable_string = (gc_row_list[0]['vertices']).replace("'", "\"")
        self.gc_dict = json.loads(json_acceptable_string)

        self.train_name_list = already_labelled
        self.test_name_list = already_labelled
        '''

        
        # stanext breed dict (contains for each name a stanext specific index)
        breed_json_path = os.path.join(STANEXT_RELATED_DATA_ROOT_DIR, 'StanExt_breed_dict_v2.json')
        self.breed_dict = self.get_breed_dict(breed_json_path, create_new_breed_json=False) 

        # load smal symmetry info
        self.sym_ids_dict = get_symmetry_indices()
        
        '''
        self.train_name_list = sorted(self.train_name_list)
        self.test_name_list = sorted(self.test_name_list)
        random.seed(4)
        random.shuffle(self.train_name_list)
        random.shuffle(self.test_name_list)
        if shorten_dataset_to is not None:
            # sometimes it is useful to have a smaller set (validation speed, debugging)
            self.train_name_list = self.train_name_list[0 : min(len(self.train_name_list), shorten_dataset_to)]
            self.test_name_list = self.test_name_list[0 : min(len(self.test_name_list), shorten_dataset_to)]
            # special case for debugging: 12 similar images
            if shorten_dataset_to == 12:
                my_sample = self.test_name_list[2]
                for ind in range(0, 12):
                    self.test_name_list[ind] = my_sample
        '''
        print('len(dataset): ' + str(self.__len__()))

        # add results for eyes, whithers and throat as obtained through anipose -> they are used
        #   as pseudo ground truth at training time.
        # self.path_anipose_out_root = os.path.join(STANEXT_RELATED_DATA_ROOT_DIR, 'animalpose_hg8_v0_results_on_StanExt')
        self.path_anipose_out_root = os.path.join(STANEXT_RELATED_DATA_ROOT_DIR, 'animalpose_hg8_v1_results_on_StanExt')     # this is from hg_anipose_after01bugfix_v1
        # self.prepare_anipose_res_and_save()

        
    def get_data_sampler_info(self):
        # for custom data sampler
        if self.is_train:
            name_list = self.train_name_list
        else:
            name_list = self.test_name_list 
        info_dict = {'name_list': name_list,
                    'stanext_breed_dict': self.breed_dict,
                    'breeds_abbrev_dict': COMPLETE_ABBREV_DICT, 
                    'breeds_summary': COMPLETE_SUMMARY_BREEDS, 
                    'breeds_sim_martix_raw': SIM_MATRIX_RAW, 
                    'breeds_sim_abbrev_inds': SIM_ABBREV_INDICES
                    }
        return info_dict

    def get_data_sampler_info_gc(self):
        # for custom data sampler
        if self.is_train:
            name_list = self.train_name_list
        else:
            name_list = self.test_name_list 
        info_dict_gc = {'name_list': name_list,
                        'gc_annots_categories': self.gc_annots_categories,
                        }
        if self.add_nonflat:
            if self.is_train:
                name_list_nonflat = self.train_name_list_nonflat
            else:
                name_list_nonflat = self.test_name_list_nonflat
            info_dict_gc['name_list_nonflat'] = name_list_nonflat
        return info_dict_gc



    def get_breed_dict(self, breed_json_path, create_new_breed_json=False):
        if create_new_breed_json:
            breed_dict = {}
            breed_index = 0
            for img_name in self.train_name_list: 
                folder_name = img_name.split('/')[0]
                breed_name = folder_name.split(folder_name.split('-')[0] + '-')[1]
                if not (folder_name in breed_dict):
                    breed_dict[folder_name] = {
                        'breed_name': breed_name,
                        'index': breed_index}
                    breed_index += 1
            with open(breed_json_path, 'w', encoding='utf-8') as f: json.dump(breed_dict, f, ensure_ascii=False, indent=4)
        else:
            with open(breed_json_path) as json_file: breed_dict = json.load(json_file)
        return breed_dict



    def prepare_anipose_res_and_save(self):
        # I only had to run this once ...
        # path_animalpose_res_root = '/ps/scratch/nrueegg/new_projects/Animals/dog_project/pytorch-stacked-hourglass/results/animalpose_hg8_v0/'
        path_animalpose_res_root = '/is/cluster/work/nrueegg/icon_pifu_related/barc_for_bite/results/results/hg_anipose_after01bugfix_v1/stanext24_XXX_e300_json/'

        train_dict, init_test_dict, init_val_dict = utils_stanext.load_stanext_json_as_dict(split_train_test=True, V12=self.V12)
        train_name_list = list(train_dict.keys())
        val_name_list = list(init_val_dict.keys())
        test_name_list = list(init_test_dict.keys())
        all_dicts = [train_dict, init_val_dict, init_test_dict]
        all_name_lists = [train_name_list, val_name_list, test_name_list]
        all_prefixes = ['train', 'val', 'test']
        for ind in range(3):
            this_name_list = all_name_lists[ind]
            this_dict = all_dicts[ind]
            this_prefix = all_prefixes[ind]

            for index in range(0, len(this_name_list)):
                print(index)
                name = this_name_list[index]
                data = this_dict[name]

                img_path = os.path.join(self.img_folder, data['img_path'])

                path_animalpose_res = os.path.join(path_animalpose_res_root.replace('XXX', this_prefix), data['img_path'].replace('.jpg', '.json'))


                # prepare predicted keypoints
                '''if is_train:
                    path_animalpose_res = os.path.join(path_animalpose_res_root, 'train_stanext', 'res_' + str(index) + '.json')
                else:
                    path_animalpose_res = os.path.join(path_animalpose_res_root, 'test_stanext', 'res_' + str(index) + '.json')
                '''
                with open(path_animalpose_res) as f: animalpose_data = json.load(f)
                anipose_joints_256 = np.asarray(animalpose_data['pred_joints_256']).reshape((-1, 3)) 
                anipose_center = animalpose_data['center']
                anipose_scale = animalpose_data['scale']
                anipose_joints_64 = anipose_joints_256 / 4        
                '''thrs_21to24 = 0.2
                anipose_joints_21to24 = np.zeros((4, 3)))    
                for ind_j in range(0:4):
                    anipose_joints_untrans = transform(anipose_joints_64[20+ind_j, 0:2], anipose_center, anipose_scale, [64, 64], invert=True, rot=0, as_int=False)-1
                    anipose_joints_trans_again = transform(anipose_joints_untrans+1, anipose_center, anipose_scale, [64, 64], invert=False, rot=0, as_int=False)
                    anipose_joints_21to24[ind_j, :2] = anipose_joints_untrans
                    if anipose_joints_256[20+ind_j, 2] >= thrs_21to24:
                        anipose_joints_21to24[ind_j, 2] = 1'''
                anipose_joints_0to24 = np.zeros((24, 3))
                for ind_j in range(24):
                    # anipose_joints_untrans = transform(anipose_joints_64[ind_j, 0:2], anipose_center, anipose_scale, [64, 64], invert=True, rot=0, as_int=False)-1
                    anipose_joints_untrans = transform(anipose_joints_64[ind_j, 0:2]+1, anipose_center, anipose_scale, [64, 64], invert=True, rot=0, as_int=False)-1
                    anipose_joints_0to24[ind_j, :2] = anipose_joints_untrans
                    anipose_joints_0to24[ind_j, 2] = anipose_joints_256[ind_j, 2]
                # save anipose result for usage later on
                out_path = os.path.join(self.path_anipose_out_root, data['img_path'].replace('.jpg', '.json'))
                if not os.path.exists(os.path.dirname(out_path)): os.makedirs(os.path.dirname(out_path))
                out_dict = {'orig_anipose_joints_256': list(anipose_joints_256.reshape((-1))),
                            'anipose_joints_0to24': list(anipose_joints_0to24[:, :3].reshape((-1))),
                            'orig_index': index,
                            'orig_scale': animalpose_data['scale'],
                            'orig_center': animalpose_data['center'],
                            'data_split': this_prefix,      # 'is_train': is_train, 
                            }
                with open(out_path, 'w') as outfile: json.dump(out_dict, outfile)
        return
















    def __getitem__(self, index):

        
        if self.is_train:
            train_val_test_Prefix = 'train'
            if self.add_nonflat and index >= len(self.train_name_list):
                name = self.train_name_list_nonflat[index - len(self.train_name_list)]
                gc_isflat = 0
            else:
                name = self.train_name_list[index]
                gc_isflat = 1
            data = self.train_dict[name]
        else:
            train_val_test_Prefix = self.val_opt        # 'val' or 'test'
            if self.add_nonflat and index >= len(self.test_name_list):
                name = self.test_name_list_nonflat[index - len(self.test_name_list)]
                gc_isflat = 0
            else:
                name = self.test_name_list[index]
                gc_isflat = 1
            data = self.test_dict[name]
        img_path = os.path.join(self.img_folder, data['img_path'])


        '''
        # for debugging only
        train_val_test_Prefix = 'train'
        name = self.train_name_list[index]
        data = self.trainvaltest_dict[name]
        img_path = os.path.join(self.img_folder, data['img_path'])

        if self.dataset_mode=='complete_with_gc':
            n_verts_smal = 3889

            gc_info_raw = self.gc_dict['bite/' + name]      # a list with all vertex numbers that are in ground contact
            gc_info = []
            gc_info_tch = torch.zeros((n_verts_smal))
            for ind_v in gc_info_raw:
                if ind_v < n_verts_smal:
                    gc_info.append(ind_v)
                    gc_info_tch[ind_v] = 1
            gc_info_available = True
        '''

        # array of shape (n_verts_smal, 3) with [first: no-contact=0 contact=1     second: index of vertex     third: dist]
        n_verts_smal = 3889
        if gc_isflat:
            if name.split('.')[0] in self.gc_annots_overview_stage3:
                gc_vertdists_overview = self.gc_annots_overview_stage3[name.split('.')[0]]['gc_vertdists_overview']
                gc_info_tch = torch.tensor(gc_vertdists_overview[:, :]) # torch.tensor(gc_vertdists_overview[:, 0])
                gc_info_available = True
                gc_touching_ground = True
            elif name.split('.')[0] in self.gc_annots_overview_stage2b_contact:
                gc_vertdists_overview = self.gc_annots_overview_stage2b_contact[name.split('.')[0]]['gc_vertdists_overview']
                gc_info_tch = torch.tensor(gc_vertdists_overview[:, :]) # torch.tensor(gc_vertdists_overview[:, 0])
                gc_info_available = True
                gc_touching_ground = True
            elif name.split('.')[0] in self.gc_annots_overview_stage2b_nocontact:
                gc_info_tch = torch.zeros((n_verts_smal, 3)) 
                gc_info_tch[:, 2] = 2.0     # big distance
                gc_info_available = True
                gc_touching_ground = False
            else:
                if 'pose' in self.gc_annots_categories[name]:
                    pose_label = self.gc_annots_categories[name]['pose']
                    if pose_label in ['standing_4paws']:
                        gc_vertdists_overview = self.gc_annots_overview_stages12_all4pawsincontact['all4pawsincontact']['gc_vertdists_overview']
                        gc_info_tch = torch.tensor(gc_vertdists_overview[:, :]) # torch.tensor(gc_vertdists_overview[:, 0])
                        gc_info_available = True
                        gc_touching_ground = True
                    elif pose_label in ['jumping_nottouching']:
                        gc_info_tch = torch.zeros((n_verts_smal, 3)) 
                        gc_info_tch[:, 2] = 2.0     # big distance
                        gc_info_available = True
                        gc_touching_ground = False
                    else:
                        gc_info_tch = torch.zeros((n_verts_smal, 3)) 
                        gc_info_tch[:, 2] = 2.0     # big distance
                        gc_info_available = False    
                        gc_touching_ground = False
        else:
            gc_info_tch = torch.zeros((n_verts_smal, 3)) 
            gc_info_tch[:, 2] = 2.0     # big distance
            gc_info_available = False    
            gc_touching_ground = False            


        # is this pose approximatly symmetric? head pose is not considered
        approximately_symmetric_pose = False
        if 'pose' in self.gc_annots_categories[name]:
            pose_label = self.gc_annots_categories[name]['pose']
            if pose_label in ['lying_sym', 'sitting_sym']:
                approximately_symmetric_pose = True




        # import pdb; pdb.set_trace()
        debugging = False
        if debugging:
            import shutil
            import trimesh
            from smal_pytorch.smal_model.smal_torch_new import SMAL
            smal = SMAL()
            verts = smal.v_template.detach().cpu().numpy()
            faces = smal.faces.detach().cpu().numpy()
            vert_colors = np.repeat(255*gc_info_tch[:, 0].detach().cpu().numpy()[:, None], 3, 1)
            # vert_colors = np.repeat(255*gc_info_np[:, None], 3, 1)
            my_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False,  maintain_order=True)
            my_mesh.visual.vertex_colors = vert_colors
            debug_folder = '/is/cluster/work/nrueegg/icon_pifu_related/barc_for_bite/debugging/gc_debugging/'
            my_mesh.export(debug_folder + (name.split('/')[1]).replace('.jpg', '_withgc.obj'))
            shutil.copy(img_path, debug_folder + name.split('/')[1])




        
        sf = self.scale_factor
        rf = self.rot_factor
        try:
            # import pdb; pdb.set_trace()

            '''new_anipose_root_path = '/is/cluster/work/nrueegg/icon_pifu_related/barc_for_bite/results/results/hg_anipose_after01bugfix_v1/stanext24_XXX_e300_json/'
            adjusted_new_anipose_root_path = new_anipose_root_path.replace('XXX', train_val_test_Prefix)
            new_anipose_res_path = adjusted_new_anipose_root_path + data['img_path'].replace('.jpg', '.json')
            with open(new_anipose_res_path) as f: new_anipose_data = json.load(f)
            '''

            anipose_res_path = os.path.join(self.path_anipose_out_root, data['img_path'].replace('.jpg', '.json'))
            with open(anipose_res_path) as f: anipose_data = json.load(f)
            anipose_thr = 0.2
            anipose_joints_0to24 = np.asarray(anipose_data['anipose_joints_0to24']).reshape((-1, 3))
            anipose_joints_0to24_scores = anipose_joints_0to24[:, 2]
            # anipose_joints_0to24_scores[anipose_joints_0to24_scores>anipose_thr] = 1.0
            anipose_joints_0to24_scores[anipose_joints_0to24_scores<anipose_thr] = 0.0
            anipose_joints_0to24[:, 2] = anipose_joints_0to24_scores
        except:
            # REMARK: This happens sometimes!!! maybe once every 10th image..?
            # print('no anipose eye keypoints!')
            anipose_joints_0to24 = np.zeros((24, 3))

        joints = np.concatenate((np.asarray(data['joints'])[:20, :], anipose_joints_0to24[20:24, :]), axis=0)
        joints[joints[:, 2]==0, :2] = 0     # avoid nan values
        pts = torch.Tensor(joints)

        # inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        # sf = scale * 200.0 / res[0]  # res[0]=256
        # center = center * 1.0 / sf
        # scale = scale / sf = 256 / 200
        # h = 200 * scale
        bbox_xywh = data['img_bbox']
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
            seg = torch.Tensor(utils_stanext.get_seg_from_entry(data)[None, :, :])
            seg = torch.cat(3*[seg])

        r = 0
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
                pts = shufflelr(pts, img.size(2), self.DATA_INFO.hflip_indices)
                c[0] = img.size(2) - c[0]
                # flip ground contact annotations
                gc_info_tch_swapped = torch.zeros_like(gc_info_tch)
                gc_info_tch_swapped[self.sym_ids_dict['center'], :] = gc_info_tch[self.sym_ids_dict['center'], :]
                gc_info_tch_swapped[self.sym_ids_dict['right'], :] = gc_info_tch[self.sym_ids_dict['left'], :]
                gc_info_tch_swapped[self.sym_ids_dict['left'], :] = gc_info_tch[self.sym_ids_dict['right'], :]
                gc_info_tch = gc_info_tch_swapped


            # Color
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)




        # import pdb; pdb.set_trace()
        debugging = False
        if debugging and do_flip:
            import shutil
            import trimesh
            from smal_pytorch.smal_model.smal_torch_new import SMAL
            smal = SMAL()
            verts = smal.v_template.detach().cpu().numpy()
            faces = smal.faces.detach().cpu().numpy()
            vert_colors = np.repeat(255*gc_info_tch[:, 0].detach().cpu().numpy()[:, None], 3, 1)
            # vert_colors = np.repeat(255*gc_info_np[:, None], 3, 1)
            my_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False,  maintain_order=True)
            my_mesh.visual.vertex_colors = vert_colors
            debug_folder = '/is/cluster/work/nrueegg/icon_pifu_related/barc_for_bite/debugging/gc_debugging/'
            my_mesh.export(debug_folder + (name.split('/')[1]).replace('.jpg', '_withgc_flip.obj'))














        # Prepare image and groundtruth map
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        img_border_mask = torch.all(inp > 1.0/256, dim = 0).unsqueeze(0).float()        # 1 is foreground
        inp = color_normalize(inp, self.DATA_INFO.rgb_mean, self.DATA_INFO.rgb_stddev)
        if self.calc_seg:
            seg = crop(seg, c, s, [self.inp_res, self.inp_res], rot=r)

        # Generate ground truth
        tpts = pts.clone()
        target_weight = tpts[:, 2].clone().view(nparts, 1)
        
        target = torch.zeros(nparts, self.out_res, self.out_res)
        for i in range(nparts):
            # if tpts[i, 2] > 0: # This is evil!!
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2]+1, c, s, [self.out_res, self.out_res], rot=r, as_int=False)) - 1
                target[i], vis = draw_labelmap(target[i], tpts[i], self.sigma, type=self.label_type)
                target_weight[i, 0] *= vis
        # NEW:
        '''target_new, vis_new = draw_multiple_labelmaps((self.out_res, self.out_res), tpts[:, :2]-1, self.sigma, type=self.label_type)
        target_weight_new = tpts[:, 2].clone().view(nparts, 1) * vis_new
        target_new[(target_weight_new==0).reshape((-1)), :, :] = 0'''

                        
        # --- Meta info
        this_breed = self.breed_dict[name.split('/')[0]]        # 120
        # add information about location within breed similarity matrix
        folder_name = name.split('/')[0]
        breed_name = folder_name.split(folder_name.split('-')[0] + '-')[1]
        abbrev = COMPLETE_ABBREV_DICT[breed_name]
        try:
            sim_breed_index = COMPLETE_SUMMARY_BREEDS[abbrev]._ind_in_xlsx_matrix 
        except: # some breeds are not in the xlsx file
            sim_breed_index = -1
        meta = {'index' : index, 'center' : c, 'scale' : s,
            'pts' : pts, 'tpts' : tpts, 'target_weight': target_weight, 
            'breed_index': this_breed['index'], 'sim_breed_index': sim_breed_index,
            'ind_dataset': 0}   # ind_dataset=0 for stanext or stanexteasy or stanext 2
        meta2 = {'index' : index, 'center' : c, 'scale' : s,
            'pts' : pts, 'tpts' : tpts, 'target_weight': target_weight, 
           'ind_dataset': 3} 

        # import pdb; pdb.set_trace()

        # out_path_root = '/is/cluster/work/nrueegg/icon_pifu_related/barc_for_bite/debugging/stanext_preprocessing/old_animalpose_version/'
        # out_path_root = '/is/cluster/work/nrueegg/icon_pifu_related/barc_for_bite/debugging/stanext_preprocessing/v0/'
        # save_input_image_with_keypoints(inp, meta['tpts'], out_path = out_path_root + name.replace('/', '_'), ratio_in_out=self.inp_res/self.out_res)


        # return different things depending on dataset_mode
        if self.dataset_mode=='keyp_only':
            # save_input_image_with_keypoints(inp, meta['tpts'], out_path='./test_input_stanext.png', ratio_in_out=self.inp_res/self.out_res)
            return inp, target, meta
        elif self.dataset_mode=='keyp_and_seg':
            meta['silh'] = seg[0, :, :]
            meta['name'] = name
            return inp, target, meta
        elif self.dataset_mode=='keyp_and_seg_and_partseg':
            # partseg is fake! this does only exist such that this dataset can be combined with an other datset that has part segmentations
            meta2['silh'] = seg[0, :, :]
            meta2['name'] = name
            fake_body_part_matrix = torch.ones((3, 256, 256)).long() * (-1)
            meta2['body_part_matrix'] = fake_body_part_matrix
            return inp, target, meta2
        elif (self.dataset_mode=='complete') or (self.dataset_mode=='complete_with_gc'):
            target_dict = meta
            target_dict['silh'] = seg[0, :, :]
            # NEW for silhouette loss
            target_dict['img_border_mask'] = img_border_mask
            target_dict['has_seg'] = True
            # ground contact
            if self.dataset_mode=='complete_with_gc':
                target_dict['has_gc_is_touching'] = gc_touching_ground
                target_dict['has_gc'] = gc_info_available 
                target_dict['gc'] = gc_info_tch
                target_dict['approximately_symmetric_pose'] = approximately_symmetric_pose
                target_dict['isflat'] = gc_isflat
            if target_dict['silh'].sum() < 1:
                if  ((not self.is_train) and self.val_opt == 'test'):
                    raise ValueError
                elif self.is_train:
                    print('had to replace training image')
                    replacement_index = max(0, index - 1)
                    inp, target_dict = self.__getitem__(replacement_index)                    
                else:
                    # There seem to be a few validation images without segmentation
                    # which would lead to nan in iou calculation
                    replacement_index = max(0, index - 1)
                    inp, target_dict = self.__getitem__(replacement_index)
            return inp, target_dict
        else:
            print('sampling error')
            import pdb; pdb.set_trace()
            raise ValueError

    def get_len_nonflat(self):
        if self.is_train:
            return len(self.train_name_list_nonflat) 
        else:
            return len(self.test_name_list_nonflat)  


    def __len__(self):
        if self.is_train:
            return len(self.train_name_list) 
        else:
            return len(self.test_name_list)   


