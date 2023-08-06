
import numpy as np
import random
import copy
import time
import warnings
import random

from torch.utils.data import Sampler
from torch._six import int_classes as _int_classes

class CustomGCSamplerNoCLass(Sampler):
    """Wraps another sampler to yield a mini-batch of indices.
    The structure of this sampler is way to complicated because it is a shorter/simplified version of 
    CustomBatchSampler. The relations between breeds are not relevant for the cvpr 2022 paper, but we kept 
    this structure which we were using for the experiments with clade related losses. ToDo: restructure 
    this sampler. 
    Args:
        data_sampler_info (dict): a dictionnary, containing information about the dataset and breeds. 
        batch_size (int): Size of mini-batch.
    """

    def __init__(self, data_sampler_info_gc, batch_size, add_nonflat=False):
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            assert (batch_size == 12 and add_nonflat==False) or (batch_size == 14 and add_nonflat==True) 
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        self.data_sampler_info_gc = data_sampler_info_gc
        self.batch_size = batch_size
        self.add_nonflat = add_nonflat

        self.n_images_tot = len(self.data_sampler_info_gc['name_list'])  # 4305

        # get full sorted image list
        self.pose_dict = {}
        self.dict_name_to_idx = {}
        for ind_img, img in enumerate(self.data_sampler_info_gc['name_list']):
            self.dict_name_to_idx[img] = ind_img
            pose = self.data_sampler_info_gc['gc_annots_categories'][img]['pose']
            if pose in self.pose_dict.keys():
                self.pose_dict[pose].append(img) 
            else:
                self.pose_dict[pose] = [img]

        # prepare non-flat images
        if self.add_nonflat:
            self.n_images_nonflat_tot = len(self.data_sampler_info_gc['name_list_nonflat'])

        # self.n_desired_batches = int(np.floor(len(self.data_sampler_info_gc['name_list']) / batch_size))        # 157
        self.n_desired_batches = 160 

    def get_description(self):
        description = "\
            This sampler returns stanext data such that poses are more balanced. \n\
            -> works on top of stanext24_withgc_v2"
        return description

    def get_nonflat_idx_list(self, shuffle=True):
        all_nonflat_idxs = list(range(self.n_images_tot, self.n_images_tot + self.n_images_nonflat_tot))
        if shuffle:
            random.shuffle(all_nonflat_idxs)
        return all_nonflat_idxs

    def get_list_for_group_index(self, ind_g, n_groups=1, shuffle=True, return_info=False):
        # availabe poses
        #   sitting_sym: 561
        #   lying_sym: 199
        #   jumping_touching: 21
        #   standing_4paws: 1999
        #   running: 132
        #   sitting_comp: 306
        #   onhindlegs: 16
        #   walking: 325
        #   lying_comp: 596
        #   standing_fewpaws: 98
        #   otherpose: 22
        #   downwardfacingdog: 14
        #   jumping_nottouching: 16
        # 
        # available groups (7 groups)
        #   89: 'otherpose', 'downwardfacingdog', 'jumping_nottouching', 'onhindlegs', 'jumping_touching' 
        #   561: 'sitting_sym'
        #   306: 'sitting_comp'
        #   199: 'lying_sym'
        #   596: 'lying_comp'
        #   555: 'standing_fewpaws', 'running', 'walking'
        #   1999: 'standing_4paws'
        #       -> sample: 2, 1.5, 1.5, 1.5, 1.5, 2, 2
        # 
        # available groups (5 groups)
        #   89: 'otherpose', 'downwardfacingdog', 'jumping_nottouching', 'onhindlegs', 'jumping_touching' 
        #   867: 'sitting_sym', 'sitting_comp'
        #   795: 'lying_sym', 'lying_comp'
        #   555: 'standing_fewpaws', 'running', 'walking'
        #   1999: 'standing_4paws'
        #       -> sample: 2, 3, 3, 2, 2
        assert (n_groups == 1)
        if ind_g == 0:
            n_samples_per_batch = 12
            pose_names = ['otherpose', 'downwardfacingdog', 'jumping_nottouching', 'onhindlegs', 'jumping_touching', 'sitting_sym', 'sitting_comp', 'lying_sym', 'lying_comp', 'standing_fewpaws', 'running', 'walking', 'standing_4paws']
        all_imgs_this_group = []
        for pose_name in pose_names:
            all_imgs_this_group.extend(self.pose_dict[pose_name])
        if shuffle:
            random.shuffle(all_imgs_this_group)
        if return_info:
            return all_imgs_this_group, pose_names, n_samples_per_batch
        else:
            return all_imgs_this_group


    def __iter__(self):

        n_groups = 1
        group_lists = {}
        n_samples_per_batch = {}
        for ind_g in range(n_groups):
            group_lists[ind_g], pose_names, n_samples_per_batch[ind_g] = self.get_list_for_group_index(ind_g, n_groups=1, shuffle=True, return_info=True)
        if self.add_nonflat:
            nonflat_idx_list = self.get_nonflat_idx_list()

        # we want to sample all sitting poses at least once per batch (and ths all other 
        #   images except standing on 4 paws)
        all_batches = []
        for ind in range(self.n_desired_batches):
            batch_with_idxs = []
            for ind_g in range(n_groups):
                for ind_s in range(n_samples_per_batch[ind_g]):
                    if len(group_lists[ind_g]) == 0:
                        group_lists[ind_g] = self.get_list_for_group_index(ind_g, n_groups=1, shuffle=True)
                    name = group_lists[ind_g].pop(0)
                    idx = self.dict_name_to_idx[name]
                    batch_with_idxs.append(idx)
            if self.add_nonflat:
                for ind_x in range(2):
                    if len(nonflat_idx_list) == 0:
                        nonflat_idx_list = self.get_nonflat_idx_list()
                    idx = nonflat_idx_list.pop(0)
                    batch_with_idxs.append(idx)
            all_batches.append(batch_with_idxs)

        for batch in all_batches:
            yield batch


    def __len__(self):
        # Since we are sampling pairs of dogs and not each breed has an even number of dogs, we can not 
        # guarantee to show each dog exacly once. What we do instead, is returning the same amount of 
        # batches as we would return with a standard sampler which is not based on dog pairs.    
        '''if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore'''
        return self.n_desired_batches








