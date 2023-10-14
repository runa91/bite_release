

import os
import glob
import csv
import numpy as np
import cv2
import math
import glob
import pickle as pkl
import open3d as o3d
import trimesh
import torch
import torch.utils.data as data

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.anipose_data_info import COMPLETE_DATA_INFO        
from stacked_hourglass.utils.imutils import load_image 
from stacked_hourglass.utils.transforms import crop, color_normalize
from stacked_hourglass.utils.pilutil import imresize 
from stacked_hourglass.utils.imutils import im_to_torch
from configs.dataset_path_configs import TEST_IMAGE_CROP_ROOT_DIR
from configs.data_info import COMPLETE_DATA_INFO_24


class SketchfabScans(data.Dataset):
    DATA_INFO = COMPLETE_DATA_INFO_24
    ACC_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16]  

    def __init__(self, img_crop_folder='default', image_path=None, is_train=False, inp_res=256, out_res=64, sigma=1,
                 scale_factor=0.25, rot_factor=30, label_type='Gaussian', 
                 do_augment='default', shorten_dataset_to=None, dataset_mode='keyp_only'):
        assert is_train == False
        assert do_augment == 'default' or do_augment == False
        self.inp_res = inp_res

        self.n_pcpoints = 3000
        self.folder_imgs = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'datasets', 'sketchfab_test_set', 'images')
        self.folder_silh = self.folder_imgs.replace('images', 'silhouettes')
        self.folder_point_clouds = self.folder_imgs.replace('images', 'point_clouds_' + str(self.n_pcpoints))
        self.folder_meshes = self.folder_imgs.replace('images', 'meshes')
        self.csv_keyp_annots_path = self.folder_imgs.replace('images', 'keypoint_annotations/sketchfab_joint_annotations_complete.csv')
        self.pkl_keyp_annots_path = self.folder_imgs.replace('images', 'keypoint_annotations/sketchfab_joint_annotations_complete_but_as_pkl_file.pkl')
        self.all_mesh_paths = glob.glob(self.folder_meshes + '/**/*.obj', recursive=True)
        name_list = glob.glob(os.path.join(self.folder_imgs, '*.png')) + glob.glob(os.path.join(self.folder_imgs, '*.jpg')) + glob.glob(os.path.join(self.folder_imgs, '*.jpeg'))
        name_list = sorted(name_list)
        self.test_name_list = []
        for name in name_list:
            self.test_name_list.append(name.split('/')[-1])

        print('len(dataset): ' + str(self.__len__()))
        
        self.test_mesh_path_list = []
        self.all_pc_paths = []
        for index in range(len(self.test_name_list)):
            img_name = self.test_name_list[index]
            dog_name = img_name.split('_' + img_name.split('_')[-1])[0]
            breed = img_name.split('_')[0]      # will be french instead of french_bulldog
            mask = img_name.split('_')[-2]
            mesh_path = self.folder_meshes + '/' + dog_name + '.obj'
            path_pc = self.folder_point_clouds + '/' + dog_name + '.ply'
            if dog_name in  ['dalmatian_1281', 'french_bulldog_13']:
                mesh_path_for_pc = self.folder_meshes + '/' + dog_name + '_simple.obj'
            else:
                mesh_path_for_pc = mesh_path 
            self.test_mesh_path_list.append(mesh_path)
            if os.path.isfile(path_pc):
                self.all_pc_paths.append(path_pc)
            else:
                try:
                    mesh_gt = o3d.io.read_triangle_mesh(mesh_path_for_pc)
                except:
                    import pdb; pdb.set_trace()
                    mesh = trimesh.load(mesh_path_for_pc, process=False,  maintain_order=True) 
                    vertices = mesh.vertices
                    faces = mesh.faces

                print(mesh_path_for_pc)
                pointcloud = mesh_gt.sample_points_uniformly(number_of_points=self.n_pcpoints)
                o3d.io.write_point_cloud(path_pc, pointcloud, write_ascii=False, compressed=False, print_progress=False)
                self.all_pc_paths.append(path_pc)

        # add keypoint annotations (mesh vertices)
        read_annots_from_csv = False        # True
        if read_annots_from_csv:
            self.all_keypoint_annotations, self.keypoint_name_dict = self._read_keypoint_csv(self.csv_keyp_annots_path, folder_meshes=self.folder_meshes, get_keyp_coords=True)
            with open(self.pkl_keyp_annots_path, 'wb') as handle:
                pkl.dump(self.all_keypoint_annotations, handle, protocol=pkl.HIGHEST_PROTOCOL)
        else:
            with open(self.pkl_keyp_annots_path, 'rb') as handle:
                self.all_keypoint_annotations = pkl.load(handle)


    def _read_keypoint_csv(self, csv_path, folder_meshes=None, get_keyp_coords=True, visualize=False):
        with open(csv_path,'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            row_list = [{h:x for (h,x) in zip(headers,row)} for row in reader]
            assert(headers[2] == 'hiwi')
        keypoint_names = headers[3:]
        center_keypoint_names = ['nose','tail_start','tail_end']
        right_keypoint_names = ['right_front_paw','right_front_elbow','right_back_paw','right_back_hock','right_ear_top','right_ear_bottom','right_eye']
        left_keypoint_names = ['left_front_paw','left_front_elbow','left_back_paw','left_back_hock','left_ear_top','left_ear_bottom','left_eye']
        keypoint_name_dict = {'all': keypoint_names, 'left': left_keypoint_names, 'right': right_keypoint_names, 'center': center_keypoint_names}
        # prepare output dicts
        all_keypoint_annotations = {}
        for ind in range(len(row_list)):
            name = row_list[ind]['mesh_name']
            this_dict = row_list[ind]
            del this_dict['hiwi']
            all_keypoint_annotations[name] = this_dict
            keypoint_idxs = np.zeros((len(keypoint_names), 2))
            if get_keyp_coords:
                mesh_path = folder_meshes + '/' + row_list[ind]['mesh_name']
                mesh = trimesh.load(mesh_path, process=False,  maintain_order=True) 
                vertices = mesh.vertices
                keypoint_3d_locations = np.zeros((len(keypoint_names), 4))      # 1, 2, 3: coords, 4: is_valid
            for ind_kp, name_kp in enumerate(keypoint_names):
                idx = this_dict[name_kp]
                if idx in ['', 'n/a']:
                    keypoint_idxs[ind_kp, 0] = -1
                else:
                    keypoint_idxs[ind_kp, 0] = this_dict[name_kp]
                    keypoint_idxs[ind_kp, 1] = 1        # is valid
                    if get_keyp_coords:
                        keyp = vertices[int(row_list[ind][name_kp])]
                        keypoint_3d_locations[ind_kp, :3] = keyp
                        keypoint_3d_locations[ind_kp, 3] = 1
            all_keypoint_annotations[name]['all_keypoint_vertex_idxs'] = keypoint_idxs
            if get_keyp_coords:
                all_keypoint_annotations[name]['all_keypoint_coords_and_isvalid'] = keypoint_3d_locations
        # create visualizations if desired
        if visualize:
            raise NotImplementedError       # only debug path is missing
            out_path = '.... some debug path'
            red_color = np.asarray([255, 0, 0], dtype=np.uint8)
            green_color = np.asarray([0, 255, 0], dtype=np.uint8)
            blue_color = np.asarray([0, 0, 255], dtype=np.uint8)
            for ind in range(len(row_list)):
                mesh_path = folder_meshes + '/' + row_list[ind]['mesh_name']
                mesh = trimesh.load(mesh_path, process=False,  maintain_order=True)         # maintain_order is very important!!!!!
                vertices = mesh.vertices
                faces = mesh.faces
                dog_mesh_nocolor = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, maintain_order=True)
                dog_mesh_nocolor.visual.vertex_colors = np.ones_like(vertices, dtype=np.uint8) * 255
                sphere_list = [dog_mesh_nocolor]
                for keyp_name in keypoint_names:
                    if not (row_list[ind][keyp_name] == '' or row_list[ind][keyp_name] == 'n/a'):
                        keyp = vertices[int(row_list[ind][keyp_name])]
                        sphere = trimesh.primitives.Sphere(radius=0.02, center=keyp)
                        if keyp_name in right_keypoint_names:
                            colors = np.ones_like(sphere.vertices) * red_color[None, :]
                        elif keyp_name in left_keypoint_names:
                            colors = np.ones_like(sphere.vertices) * blue_color[None, :]
                        else:
                            colors = np.ones_like(sphere.vertices) * green_color[None, :]
                        sphere.visual.vertex_colors = colors  # trimesh.visual.random_color()
                        sphere_list.append(sphere)
                scene_keyp = trimesh.Scene(sphere_list)
                scene_keyp.export(out_path + os.path.basename(mesh_path).replace('.obj', '_withkeyp.obj'))
        return all_keypoint_annotations, keypoint_name_dict



    def __getitem__(self, index):
        img_name = self.test_name_list[index]
        dog_name = img_name.split('_' + img_name.split('_')[-1])[0]
        breed = img_name.split('_')[0]      # will be french instead of french_bulldog
        mask = img_name.split('_')[-2]
        mesh_path = self.test_mesh_path_list[index]
        # mesh_gt = o3d.io.read_triangle_mesh(mesh_path)

        path_pc = self.folder_point_clouds + '/' + dog_name + '.ply'
        assert path_pc in self.all_pc_paths
        pc_trimesh = trimesh.load(path_pc, process=False, maintain_order=True)
        pc_points = np.asarray(pc_trimesh.vertices)
        assert pc_points.shape[0] == self.n_pcpoints

        # get annotated 3d keypoints
        keyp_3d = self.all_keypoint_annotations[mesh_path.split('/')[-1]]['all_keypoint_coords_and_isvalid']

        # load image
        img_path = os.path.join(self.folder_imgs, img_name)
        
        img = load_image(img_path)  # CxHxW
        img_vis = np.transpose(img, (1, 2, 0))
        seg_path = os.path.join(self.folder_silh, img_name)
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)[:, :, 3]
        seg[seg>0] = 1
        seg_s0 = np.nonzero(seg.sum(axis=1)>0)[0] 
        seg_s1 = np.nonzero(seg.sum(axis=0)>0)[0] 
        bbox_xywh = [seg_s1.min(), seg_s0.min(), seg_s1.max() - seg_s1.min(), seg_s0.max() - seg_s0.min()]
        bbox_c = [bbox_xywh[0]+0.5*bbox_xywh[2], bbox_xywh[1]+0.5*bbox_xywh[3]]
        bbox_max = max(bbox_xywh[2], bbox_xywh[3])
        bbox_diag = math.sqrt(bbox_xywh[2]**2 + bbox_xywh[3]**2)
        # bbox_s = bbox_max / 200.      # the dog will fill the image -> bbox_max = 256
        # bbox_s = bbox_diag / 200.     # diagonal of the boundingbox will be 200
        bbox_s = bbox_max / 200. * 256. / 200.  # maximum side of the bbox will be 200
        c = torch.Tensor(bbox_c)
        s = bbox_s
        r = 0

        # Prepare image and groundtruth map
        inp_col = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = color_normalize(inp_col, self.DATA_INFO.rgb_mean, self.DATA_INFO.rgb_stddev)

        silh_3channels = np.stack((seg, seg, seg), axis=0)
        inp_silh = crop(silh_3channels, c, s, [self.inp_res, self.inp_res], rot=r)

        # add the following fields to make it compatible with stanext, most of them are fake
        target_dict = {'index': index, 'center' : -2, 'scale' : -2, 
            'breed_index': -2, 'sim_breed_index': -2,
            'ind_dataset': 1}
        target_dict['pts'] = np.zeros((self.DATA_INFO.n_keyp, 3))
        target_dict['tpts'] = np.zeros((self.DATA_INFO.n_keyp, 3))
        target_dict['target_weight'] = np.zeros((self.DATA_INFO.n_keyp, 1))
        target_dict['silh'] = inp_silh[0, :, :]      # np.zeros((self.inp_res, self.inp_res))
        target_dict['mesh_path'] = mesh_path
        target_dict['pointcloud_path'] = path_pc
        target_dict['pointcloud_points'] = pc_points
        target_dict['keypoints_3d'] = keyp_3d
        return inp, target_dict


    def __len__(self):
        return len(self.test_name_list)   









