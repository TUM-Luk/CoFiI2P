import numpy as np
import math
import torch

class Options_CARLA:
    def __init__(self):
        # data config
        self.dataroot = './dataset_large_int_train'
        self.train_subdir = 'mapping'
        self.val_subdir = 'query'
        self.test_subdir = 'query'
        
        self.train_txt = "dataset_large_int_train/train_list_deepi2p/train_75scene_t3_int4.txt"
        self.val_txt = "dataset_large_int_train/train_list_deepi2p/val_75scene_t3_int4.txt"
        self.test_txt = "dataset_large_int_train/train_list_deepi2p/val_75scene_t10_int1.txt"
        self.pin_memory = True
        
        self.vis_debug = False
        
        self.epoch = 25
        self.data_path = "./kitti_data"
        self.root_path = '.'
        self.save_path = "checkpoints"
        self.log_path = "logs"
        self.accumulation_frame_num = 3
        self.accumulation_frame_skip = 6

        self.crop_original_top_rows = 50
        self.img_scale = 1/3.75
        self.img_H = 288 
        self.img_W = 512  
        self.img_fine_resolution_scale = 32

        self.num_pc = 20480
        self.num_kpt = 64
        self.pc_min_range = -1.0
        self.pc_max_range = 40.0 # 80 for kitti in cofii2p
        self.node_a_num = 1280
        self.node_b_num = 1280
        self.k_ab = 16
        self.k_interp_ab = 3
        self.k_interp_point_a = 3
        self.k_interp_point_b = 3

        # CAM coordinate
        self.P_tx_amplitude = 0
        self.P_ty_amplitude = 0
        self.P_tz_amplitude = 0
        self.P_Rx_amplitude = 0.0 * math.pi / 12.0
        self.P_Ry_amplitude = 0.0 * math.pi / 12.0
        self.P_Rz_amplitude = 2.0 * math.pi
        self.dist_thres = 1.0
        self.img_thres = 0.9
        self.pc_thres = 0.9
        self.pos_margin = 0.2
        self.neg_margin = 1.8

        self.train_batch_size = 1
        self.val_batch_size = 1
        self.dataloader_threads = 4
        self.gpu_ids = [0]
        self.device = torch.device('cuda', self.gpu_ids[0])
        self.norm = 'gn'
        self.group_norm = 32
        self.norm_momentum = 0.1
        self.activation = 'relu'
        self.lr = 1e-3
        self.min_lr = 1e-5
        self.lr_decay_step = 10
        self.lr_decay_scale = 0.5
        self.val_freq = 2500

# 