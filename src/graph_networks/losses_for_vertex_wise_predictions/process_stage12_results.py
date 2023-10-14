
# see also (laptop):
#   /home/nadine/Documents/PhD/icon_barc_project/AMT_ground_contact_studies/stages_1and2_together/evaluate_stages12_main_forstage2b_new.py
# 
# python src/graph_networks/losses_for_vertex_wise_predictions/process_stage12_results.py
# 



import numpy as np
import os
import sys
import csv
import shutil
import pickle as pkl

ROOT_path = '/home/nadine/Documents/PhD/icon_barc_project/AMT_ground_contact_studies/'
ROOT_path_images = '/ps/scratch/nrueegg/new_projects/Animals/data/dog_datasets/Stanford_Dogs_Dataset/StanfordExtra_V12/StanExtV12_Images/'
ROOT_amt_image_list = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stages12together/amt_image_lists/'
ROOT_OUT_PATH = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stages12together/'


root_path_stage1 = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stage1/'
root_path_stage2 = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stage2/'

csv_file_stage1_pilot = root_path_stage1 + 'stage1_pilot_Batch_4841525_batch_results.csv'   
csv_file_stage1_main = root_path_stage1 + 'stage1_main_stage1_Batch_4890079_batch_results.csv'   
csv_file_stage2_pilot = root_path_stage2 + 'stage2_pilot_DogStage2PilotResults.csv'   
csv_file_stage2_main = root_path_stage2 + 'stage2_main_Batch_4890110_batch_results.csv'   

full_amt_image_list = ROOT_amt_image_list + 'all_stanext_image_names_amt.txt'
train_amt_image_list = ROOT_amt_image_list + 'all_stanext_image_names_train.txt'
test_amt_image_list = ROOT_amt_image_list + 'all_stanext_image_names_test.txt'
val_amt_image_list = ROOT_amt_image_list + 'all_stanext_image_names_val.txt'

experiment_name = 'stage_2b_image_paths'
AMT_images_root_path = 'https://dogvisground.s3.eu-central-1.amazonaws.com/StanExtV12_Images/' # n02085620-Chihuahua/n02085620_10074.jpg'
# out_folder = '/home/nadine/Documents/PhD/icon_barc_project/AMT_ground_contact_studies/stage_2b/stage2b_html_and_csv_files/'
# out_folder_imgs = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stages12together/'
# csv_out_path_pilot = out_folder + experiment_name + '_pilot_bs22.csv'
# csv_out_path_main = out_folder + experiment_name + '_main_bs22.csv'





pose_dict = {'1':'<b>Standing still</b>, all <b>four paws fully on the ground</b>',
            '2':'<b>Standing still</b>, at least one <b>paw lifted</b> (if you are in doubt if the paw is on the ground or not, choose this option)',
            '3':'<b>Walking or trotting</b> (walk, amble, pace, trot)',
            '4':'<b>Running</b> (only canter, gallup, run)',
            '5':'<b>Sitting, symmetrical legs</b>',
            '6':'<b>Sitting, complicated pose</b> (every sitting pose with asymmetrical leg position)',
            '7':'<b>lying, symmetrical legs</b> (and not lying on the side)',
            '8':'<b>lying, complicated pose</b> (every lying pose with asymmetrical leg position)',
            '9':'<b>Jumping, not touching</b> the ground',
            '10':'<b>Jumping</b> or about to jump, <b>touching the ground</b>',
            '11':'<b>On hind legs</b> (standing or walking or sitting)',
            '12':'<b>Downward facing dog</b>: standing on back legs/paws and bending over front leg',
            '13':'<b>Other poses</b>: being carried by a human, ...',
            '14':'<b>I can not see</b> the pose (please comment why: hidden, hairy, legs cut off, ...)'}

pose_dict_abbrev = {'1':'standing_4paws',
            '2':'standing_fewpaws',
            '3':'walking',
            '4':'running',
            '5':'sitting_sym',
            '6':'sitting_comp',
            '7':'lying_sym',
            '8':'lying_comp',
            '9':'jumping_nottouching',
            '10':'jumping_touching',
            '11':'onhindlegs',
            '12':'downwardfacingdog',
            '13':'otherpose',
            '14':'cantsee'}

def read_csv(csv_file):
    with open(csv_file,'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        row_list = [{h:x for (h,x) in zip(headers,row)} for row in reader]
    return row_list

def add_stage2_to_result_dict(row_list_stage2, result_info_dict):
    for ind_worker in range(len(row_list_stage2)):
        # print('------------------------------------')
        image_names = row_list_stage2[ind_worker]['Input.images'].split(';')
        all_answers_comment = row_list_stage2[ind_worker]['Answer.submitComments'].split(';')
        all_answers_dogpose = row_list_stage2[ind_worker]['Answer.submitValues'].split(';')
        for ind in range(len(image_names)):
            if 'Qualification_Tutorial_Images' in image_names[ind]:
                # print('skip tutorial images')
                pass
            else:
                img_subf = image_names[ind].split('/')[-2]
                img_name = image_names[ind].split('/')[-1]
                img_name_key = img_subf + '/' + img_name
                img_path = ROOT_path_images + img_subf + '/' + img_name
                this_img = {'img_name': img_name, 
                            'img_subf': img_subf, 
                            # 'img_path': img_path, 
                            'pose': pose_dict_abbrev[all_answers_dogpose[ind]],
                            'comment_pose': all_answers_comment[ind]}
                assert not (img_name_key in result_info_dict.keys())
                result_info_dict[img_name_key] = this_img
                '''folder_name = pose_dict_abbrev[all_answers_dogpose[ind]]
                img_name_out = img_name  # 'indw' + str(ind_worker) + '_' + img_name
                out_folder_this = out_folder_imgs + folder_name + '/' + img_name
                shutil.copyfile(img_path, out_folder_this + img_name_out)'''

def add_stage1_to_result_dict(row_list_stage1, result_info_dict):
    for ind_worker in range(len(row_list_stage1)):
        # print('------------------------------------')
        image_names = row_list_stage1[ind_worker]['Input.images'].split(';')
        all_answers_commentvis = row_list_stage1[ind_worker]['Answer.submitCommentsVisible'].split(';')
        all_answers_vis = row_list_stage1[ind_worker]['Answer.submitValuesVisible'].split(';')      # 1: visible, 2: not visible
        all_answers_commentground = row_list_stage1[ind_worker]['Answer.submitCommentsGround'].split(';')
        all_answers_ground = row_list_stage1[ind_worker]['Answer.submitValuesGround'].split(';')    # 1: flat, 2: not flat
        for ind in range(len(image_names)):
            if len(image_names[ind].split('/')) < 2:
                print('no more image in ind_worker ' + str(ind_worker))
            elif 'Qualification_Tutorial_Images' in image_names[ind]:
                # print('skip tutorial images')
                pass
            else:
                img_subf = image_names[ind].split('/')[-2]
                img_name = image_names[ind].split('/')[-1]
                img_name_key = img_subf + '/' + img_name
                img_path = ROOT_path_images + img_subf + '/' + img_name
                if all_answers_vis[ind] == '1':
                    vis = True
                elif all_answers_vis[ind] == '2':
                    vis = False
                else:
                    vis = None
                    # raise ValueError
                if all_answers_ground[ind] == '1':
                    flat = True
                elif all_answers_ground[ind] == '2':
                    flat = False
                else:
                    flat = None
                    # raise ValueError
                if img_name_key in result_info_dict.keys():
                    result_info_dict[img_name_key]['is_vis'] = vis
                    result_info_dict[img_name_key]['comment_vis'] = all_answers_commentvis[ind]
                    result_info_dict[img_name_key]['is_flat'] = flat
                    result_info_dict[img_name_key]['comment_flat'] = all_answers_commentground[ind]
                else:
                    print(img_path)
                    this_img = {'img_name': img_name, 
                                'img_subf': img_subf, 
                                # 'img_path': img_path, 
                                'is_vis': vis,
                                'comment_vis': all_answers_commentvis[ind], 
                                'is_flat': flat,
                                'comment_flat': all_answers_commentground[ind]}
                    result_info_dict[img_name_key] = this_img


    
# ------------------------------------------------------------------------------

'''
if not os.path.exists(out_folder_imgs): os.makedirs(out_folder_imgs)
for folder_name in pose_dict_abbrev.values():
    out_folder_this = out_folder_imgs + folder_name
    if not os.path.exists(out_folder_this): os.makedirs(out_folder_this)
'''



row_list_stage2_pilot = read_csv(csv_file_stage2_pilot)
row_list_stage1_pilot = read_csv(csv_file_stage1_pilot)
row_list_stage2_main = read_csv(csv_file_stage2_main)
row_list_stage1_main = read_csv(csv_file_stage1_main)

result_info_dict = {}
add_stage2_to_result_dict(row_list_stage2_pilot, result_info_dict)
add_stage2_to_result_dict(row_list_stage2_main, result_info_dict)
add_stage1_to_result_dict(row_list_stage1_pilot, result_info_dict)
add_stage1_to_result_dict(row_list_stage1_main, result_info_dict)



# initial image list: all_stanext_image_names_amt.txt
# (/home/nadine/Documents/PhD/icon_barc_project/AMT_ground_contact_studies/all_stanext_image_names_amt.txt)
# the initial image list did first contain randomly shuffeled {train + test} 
#   images and after that randomly shuffeled {val} images
# see also /is/cluster/work/nrueegg/icon_pifu_related/ICON/lib/ground_contact/create_gc_dataset/get_stanext_images_for_amt.py
# train and test: 6773 + 1703 = 8476
# val: 4062
with open(full_amt_image_list) as f: full_amt_lines = f.readlines()
with open(train_amt_image_list) as f: train_amt_lines = f.readlines()
with open(test_amt_image_list) as f: test_amt_lines = f.readlines()
with open(val_amt_image_list) as f: val_amt_lines = f.readlines()

for ind_l, line in enumerate(train_amt_lines):
    img_name_key = (line.split('/')[-2]) + '/' + (line.split('/')[-1]).split('\n')[0]
    result_info_dict[img_name_key]['split'] = 'train'
for ind_l, line in enumerate(test_amt_lines):
    img_name_key = (line.split('/')[-2]) + '/' + (line.split('/')[-1]).split('\n')[0]
    result_info_dict[img_name_key]['split'] = 'test'
for ind_l, line in enumerate(val_amt_lines):
    img_name_key = (line.split('/')[-2]) + '/' + (line.split('/')[-1]).split('\n')[0]
    result_info_dict[img_name_key]['split'] = 'val'


# we have stage 2b labels for:     
#   constraint_vis =  (res['is_vis'] in {True, None})
#   constraint_flat =  (res['is_flat'] in {True, None})    
#   constraint_pose = (res['pose'] in {'standing_fewpaws', 'walking', 'running', })        
# we have stage 3 labels for:     
#   constraint_vis =  (res['is_vis'] in {True, None})
#   constraint_flat =  (res['is_flat'] in {True, None})    
#   constraint_pose = (res['pose'] in {'sitting_sym', 'sitting_comp', 'lying_sym', 'lying_comp', 'downwardfacingdog', 'otherpose', 'jumping_touching', 'onhindlegs'})
# we have no labels for:
#   constraint_pose = (res['pose'] in {'standing_4paws', 'jumping_nottouching', 'cantsee'})


with open(ROOT_OUT_PATH + 'gc_annots_categories_stages12_complete.pkl', 'wb') as fp:
    pkl.dump(result_info_dict, fp)


import pdb; pdb.set_trace()




























# -------------------------------------------------------------------------------------------------

'''
# sort the result images.
all_pose_names = [*pose_dict_abbrev.values()]
split_list = ['train', 'test', 'val', 'traintest']
split_list_dict = {}
for split in split_list:
    nimgs_pose_dict = {}
    for pose_name in all_pose_names:
        nimgs_pose_dict[pose_name] = 0
    images_to_label = []
    for ind_l, line in enumerate(full_amt_lines):
        img_name = (line.split('/')[-1]).split('\n')[0]
        res = result_info_dict[img_name]
        if split == 'traintest':
            constraint_split = (res['split'] == 'train') or (res['split'] == 'test')      
        else:
            constraint_split = (res['split'] == split)        #  (res['split'] == 'train')
        constraint_vis =  (res['is_vis'] in {True, None})
        constraint_flat =  (res['is_flat'] in {True, None})
        # constraint_pose = (res['pose'] in {'sitting_sym', 'sitting_comp', 'lying_sym', 'lying_comp', 'downwardfacingdog', 'otherpose', 'jumping_touching', 'onhindlegs'})
        constraint_pose = (res['pose'] in {'standing_fewpaws', 'walking', 'running', })        
        
        if constraint_split * constraint_vis * constraint_flat == True:
            nimgs_pose_dict[res['pose']] += 1
            if constraint_pose:
                images_to_label.append(line)
                folder_name = 'imgsforstage2b_' + split     # 'imgsforstage3_train'
                out_folder_this = out_folder_imgs + folder_name + '/' 
                if not os.path.exists(out_folder_this): os.makedirs(out_folder_this)
                shutil.copyfile(res['img_path'], out_folder_this + img_name)
    print('------------------------------------------------------')
    print(split)
    print(nimgs_pose_dict)
    print(len(images_to_label))
    split_list_dict[split] = {'nimgs_pose_dict': nimgs_pose_dict,
    'len(images_to_label)': len(images_to_label),
    'images_to_label': images_to_label}



# create csv files:
traintest_list = split_list_dict['traintest']['images_to_label']
val_list = split_list_dict['val']['images_to_label']
complete_list = traintest_list + val_list

all_lines_refined = []
for line in complete_list:
    all_lines_refined.append(line.split('\n')[0])

import pdb; pdb.set_trace()
'''
















