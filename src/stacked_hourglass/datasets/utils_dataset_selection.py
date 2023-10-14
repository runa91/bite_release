
import torch
from torch.utils.data import DataLoader
import cv2
import glob
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))


def get_evaluation_dataset(cfg_data_dataset, cfg_data_val_opt, cfg_data_V12, cfg_optim_batch_size, args_workers, drop_last=False):
    # cfg_data_dataset = cfg.data.DATASET
    # cfg_data_val_opt = cfg.data.VAL_OPT 
    # cfg_data_V12 = cfg.data.V12
    # cfg_optim_batch_size = cfg.optim.BATCH_SIZE
    # args_workers = args.workers
    assert cfg_data_dataset in ['stanext24_easy', 'stanext24', 'stanext24_withgc', 'stanext24_withgc_big']
    assert cfg_data_val_opt in ['train', 'test', 'val']

    if cfg_data_dataset == 'stanext24_easy':
        from stacked_hourglass.datasets.stanext24_easy import StanExtEasy as StanExt 
        dataset_mode = 'complete'
    elif cfg_data_dataset == 'stanext24':
        from stacked_hourglass.datasets.stanext24 import StanExt 
        dataset_mode = 'complete'
    elif cfg_data_dataset == 'stanext24_withgc':
        from stacked_hourglass.datasets.stanext24_withgc import StanExtGC as StanExt 
        dataset_mode = 'complete_with_gc'
    elif cfg_data_dataset == 'stanext24_withgc_big':
        from stacked_hourglass.datasets.stanext24_withgc_v2 import StanExtGC as StanExt 
        dataset_mode = 'complete_with_gc'

    # Initialise the validation set dataloader
    if cfg_data_val_opt == 'test':
        val_dataset = StanExt(image_path=None, is_train=False, dataset_mode=dataset_mode, V12=cfg_data_V12, val_opt='test')
        test_name_list = val_dataset.test_name_list
    elif cfg_data_val_opt == 'val':
        val_dataset = StanExt(image_path=None, is_train=False, dataset_mode=dataset_mode, V12=cfg_data_V12, val_opt='val')
        test_name_list = val_dataset.test_name_list
    elif cfg_data_val_opt == 'train':
        val_dataset = StanExt(image_path=None, is_train=True, do_augment='no', dataset_mode=dataset_mode, V12=cfg_data_V12)
        test_name_list = val_dataset.train_name_list
    else:
        raise ValueError
    val_loader = DataLoader(val_dataset, batch_size=cfg_optim_batch_size, shuffle=False,
                            num_workers=args_workers, pin_memory=True, drop_last=drop_last) # False)  # , drop_last=True    args.batch_size
    len_val_dataset = len(val_dataset)
    stanext_data_info = StanExt.DATA_INFO
    stanext_acc_joints = StanExt.ACC_JOINTS
    return val_dataset, val_loader, len_val_dataset, test_name_list, stanext_data_info, stanext_acc_joints


def get_sketchfab_evaluation_dataset(cfg_optim_batch_size, args_workers):
    # cfg_optim_batch_size = cfg.optim.BATCH_SIZE
    # args_workers = args.workers
    from stacked_hourglass.datasets.sketchfab import SketchfabScans
    val_dataset = SketchfabScans(image_path=None, is_train=False, dataset_mode='complete')
    test_name_list = val_dataset.test_name_list
    val_loader = DataLoader(val_dataset, batch_size=cfg_optim_batch_size, shuffle=False,
                            num_workers=args_workers, pin_memory=True, drop_last=False)     # drop_last=True)
    from stacked_hourglass.datasets.stanext24 import StanExt
    len_val_dataset = len(val_dataset)
    stanext_data_info = StanExt.DATA_INFO
    stanext_acc_joints = StanExt.ACC_JOINTS
    return val_dataset, val_loader, len_val_dataset, test_name_list, stanext_data_info, stanext_acc_joints

def get_crop_evaluation_dataset(cfg_optim_batch_size, args_workers, input_folder):
    from stacked_hourglass.datasets.imgcropslist import ImgCrops
    image_list_paths = glob.glob(os.path.join(input_folder, '*.jpg')) + glob.glob(os.path.join(input_folder, '*.png'))
    image_list = []
    test_name_list = []
    for image_path in image_list_paths:
        test_name_list.append(os.path.basename(image_path).split('.')[0])
        img = cv2.imread(image_path)
        image_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    val_dataset = ImgCrops(image_list=image_list, bbox_list=None)
    val_loader = DataLoader(val_dataset, batch_size=cfg_optim_batch_size, shuffle=False,
                            num_workers=args_workers, pin_memory=True, drop_last=False)     # drop_last=True)
    from stacked_hourglass.datasets.stanext24 import StanExt
    len_val_dataset = len(val_dataset)
    stanext_data_info = StanExt.DATA_INFO
    stanext_acc_joints = StanExt.ACC_JOINTS
    return val_dataset, val_loader, len_val_dataset, test_name_list, stanext_data_info, stanext_acc_joints

def get_single_crop_dataset_from_image(input_image, bbox=None):
    from stacked_hourglass.datasets.imgcropslist import ImgCrops
    input_image_list = [input_image]
    if bbox is not None:
        input_bbox_list = [bbox]
    else:
        input_bbox_list = None
    # prepare data loader
    val_dataset = ImgCrops(image_list=input_image_list, bbox_list=input_bbox_list, dataset_mode='complete')
    test_name_list = val_dataset.test_name_list
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=False)   
    from stacked_hourglass.datasets.stanext24 import StanExt
    len_val_dataset = len(val_dataset)
    stanext_data_info = StanExt.DATA_INFO
    stanext_acc_joints = StanExt.ACC_JOINTS
    return val_dataset, val_loader, len_val_dataset, test_name_list, stanext_data_info, stanext_acc_joints


def get_norm_dict(data_info=None, device="cuda"):
    if data_info is None:
        from stacked_hourglass.datasets.stanext24 import StanExt
        data_info = StanExt.DATA_INFO
    norm_dict = {
        'pose_rot6d_mean': torch.from_numpy(data_info.pose_rot6d_mean).float().to(device),
        'trans_mean': torch.from_numpy(data_info.trans_mean).float().to(device),
        'trans_std': torch.from_numpy(data_info.trans_std).float().to(device),
        'flength_mean': torch.from_numpy(data_info.flength_mean).float().to(device),
        'flength_std': torch.from_numpy(data_info.flength_std).float().to(device)}
    return norm_dict