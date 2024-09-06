import os
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import math
import open3d
import cv2
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.sparse import coo_matrix
import time
from pathlib import Path

from model.kpconv.preprocess_data import precompute_point_cloud_stack_mode, precompute_point_cloud_cuda
from model.network import point2node
from kapture.io.csv import kapture_from_dir
import tqdm
import quaternion
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    DistributedSampler,
    RandomSampler,
    dataloader,
    Sampler
)
from data.depth_convert import dpt_3d_convert

class FarthestSampler:
    def __init__(self, dim=3):
        self.dim = dim

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=0)

    def sample(self, pts, k):
        farthest_pts = np.zeros((self.dim, k))
        farthest_pts_idx = np.zeros(k, dtype=np.int)
        init_idx = np.random.randint(len(pts))
        farthest_pts[:, 0] = pts[:, init_idx]
        farthest_pts_idx[0] = init_idx
        distances = self.calc_distances(farthest_pts[:, 0:1], pts)
        for i in range(1, k):
            idx = np.argmax(distances)
            farthest_pts[:, i] = pts[:, idx]
            farthest_pts_idx[i] = idx
            distances = np.minimum(distances, self.calc_distances(farthest_pts[:, i:i+1], pts))
        return farthest_pts, farthest_pts_idx

def make_carla_dataloader(mode, opt):
    data_root = opt.dataroot  # 'dataset_large_int_train'
    
    train_subdir = opt.train_subdir  # 'mapping'
    val_subdir = opt.val_subdir  # 'mapping'
    test_subdir = opt.test_subdir  # 'query'
    
    train_txt = opt.train_txt  # "dataset_large_int_train/train_list/train_t1_int1_v50_s25_io03_vo025.txt"
    val_txt = opt.val_txt 
    test_txt = opt.test_txt
    
    
    if mode == 'train':
        data_txt = train_txt
    elif mode == 'val':
        data_txt = val_txt
    elif mode == 'test':
        data_txt = test_txt
        
    with open(data_txt, 'r') as f:
        voxel_list = f.readlines()
        voxel_list = [voxel_name.rstrip() for voxel_name in voxel_list]
        
    kapture_datas={}
    sensor_datas={}
    input_path_datas={}
    train_list_kapture_map={}
    for train_path in voxel_list:
        # scene=os.path.dirname(os.path.dirname(train_path))
        scene=train_path.split('/')[0]
        if scene not in kapture_datas:
            if mode=='test':
                input_path=os.path.join(data_root,scene, test_subdir)
            elif mode=='train':
                input_path=os.path.join(data_root,scene, train_subdir)
            else:
                input_path=os.path.join(data_root, scene, val_subdir)
            kapture_data=kapture_from_dir(input_path)
            sensor_dict={}
            for timestep in kapture_data.records_camera:
                _sensor_dict=kapture_data.records_camera[timestep]
                for k, v in _sensor_dict.items():
                    sensor_dict[v]=(timestep, k)
            kapture_datas[scene]=kapture_data
            sensor_datas[scene]=sensor_dict
            input_path_datas[scene]=input_path
        train_list_kapture_map[train_path]=(kapture_datas[scene], sensor_datas[scene], input_path_datas[scene])
        
    datasets = []
    
    for train_path in tqdm.tqdm(voxel_list):
        kapture_data, sensor_data, input_path=train_list_kapture_map[train_path]
        one_dataset = carla_pc_img_dataset(root_path=data_root, train_path=train_path, mode=mode, opt=opt,
                                kapture_data=kapture_data, sensor_data=sensor_data, input_path=input_path)
        
        one_dataset[0]
        datasets.append(one_dataset)
        
    
    final_dataset = ConcatDataset(datasets)
    
    if mode=='train':
        dataloader = DataLoader(final_dataset, batch_size=opt.train_batch_size, shuffle=True, drop_last=True,
                                num_workers=opt.dataloader_threads, pin_memory=opt.pin_memory
                                )
    elif mode=='val' or mode=='test':
        dataloader = DataLoader(final_dataset, batch_size=opt.val_batch_size, shuffle=False,
                                num_workers=opt.dataloader_threads, pin_memory=opt.pin_memory
                                )
    else:
        raise ValueError
    
    return final_dataset, dataloader


class carla_pc_img_dataset(data.Dataset):
    def __init__(self, root_path, train_path, mode, opt,
                 kapture_data, sensor_data, input_path):
        super(carla_pc_img_dataset, self).__init__()
        for k,v in opt.__dict__.items():
            setattr(self,k,v)
            
        self.root_path = root_path  # "dataset_large_int_train"
        self.train_path = train_path  # 't1_int1'
        self.opt=opt
        self.mode = mode
        
        self.farthest_sampler = FarthestSampler(dim=3)
        
        self.sensor_dict = sensor_data
        self.kaptures = kapture_data
        self.input_path = input_path
        
        self.dataset = self.make_carla_dataset(root_path, train_path, mode)
        self.voxel_points = self.make_voxel_pcd()
        
        self.projector = dpt_3d_convert()
        
        print(f'load data complete. {len(self.dataset)} image-voxel pair')

    def make_carla_dataset(self, root_path, train_path, mode):
        dataset = []

        if mode == "train":
            dataset = list(self.sensor_dict.keys())
        elif mode == "val" or mode == "test":
            dataset = list(self.sensor_dict.keys())
        else:
            raise ValueError
        
        return dataset
    
    def make_voxel_pcd(self):
        scene_name = self.train_path.split('/')[0]
        point_cloud_file = os.path.join(self.input_path, f'pcd_{scene_name}_train_down.ply')
        print(f"load pcd file from {point_cloud_file}")
        pcd = open3d.io.read_point_cloud(point_cloud_file)
        pcd_points = np.array(pcd.points)
        pcd_points = pcd_points.astype(np.float32)

        voxel_points = pcd_points.T  # 整个场景（路口）的点云
        
        return voxel_points


    def downsample_with_intensity_sn(self, pointcloud, intensity, sn, voxel_grid_downsample_size):
        pcd=open3d.geometry.PointCloud()
        pcd.points=open3d.utility.Vector3dVector(np.transpose(pointcloud))
        intensity_max=np.max(intensity)

        fake_colors=np.zeros((pointcloud.shape[1],3))
        if intensity_max == 0:
            fake_colors[:, 0:1] = np.transpose(intensity)
        else:
            fake_colors[:, 0:1] = np.transpose(intensity)/intensity_max

        pcd.colors=open3d.utility.Vector3dVector(fake_colors)
        pcd.normals=open3d.utility.Vector3dVector(np.transpose(sn))

        down_pcd=pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
        down_pcd_points=np.transpose(np.asarray(down_pcd.points))
        pointcloud=down_pcd_points

        intensity=np.transpose(np.asarray(down_pcd.colors)[:,0:1])*intensity_max
        sn=np.transpose(np.asarray(down_pcd.normals))

        return pointcloud, intensity, sn

    def downsample_np(self, pc_np, intensity_np, sn_np, num_pc):
        if pc_np.shape[1] >= self.num_pc:
            choice_idx = np.random.choice(pc_np.shape[1], self.num_pc, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < self.num_pc:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], self.num_pc - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]
        intensity_np = intensity_np[:, choice_idx]
        sn_np=sn_np[:,choice_idx]
        return pc_np, intensity_np, sn_np

    def camera_matrix_cropping(self, K: np.ndarray, dx: float, dy: float):
        K_crop = np.copy(K)
        K_crop[0, 2] -= dx
        K_crop[1, 2] -= dy
        return K_crop

    def camera_matrix_scaling(self, K: np.ndarray, s: float):
        K_scale = s * K
        K_scale[2, 2] = 1
        return K_scale

    def augment_img(self, img_np):
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter(
            brightness, contrast, saturation, hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(img_np)))

        return img_color_aug_np

    def angles2rotation_matrix(self, angles):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R

    def generate_random_transform(self):
        """
        :param pc_np: pc in NWU coordinate
        :return:
        """
        t = [random.uniform(-self.P_tx_amplitude, self.P_tx_amplitude),
             random.uniform(-self.P_ty_amplitude, self.P_ty_amplitude),
             random.uniform(-self.P_tz_amplitude, self.P_tz_amplitude)]
        angles = [random.uniform(-self.P_Rx_amplitude, self.P_Rx_amplitude),
                  random.uniform(-self.P_Ry_amplitude, self.P_Ry_amplitude),
                  random.uniform(-self.P_Rz_amplitude, self.P_Rz_amplitude)]

        rotation_mat = self.angles2rotation_matrix(angles)
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        # print('t',t)
        # print('angles',angles)

        return P_random
    
    def search_point_index(self, source_points, target_points):
        '''
        source_points: [M, 3]
        target_points: [N, 3]
        '''
        indices = []
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(source_points)
        source_kdtree = open3d.geometry.KDTreeFlann(pcd)
        for i in range(target_points.shape[0]):
            [_, index, _] = source_kdtree.search_knn_vector_3d(target_points[i], 1)
        # indices = torch.nonzero(torch.isin(source_points, target_points).all(dim=1))[:, 0]
            indices.append(index)
        # print(indices.shape)
        return np.array(indices)

    def __len__(self):
        return len(self.dataset)
    
    def load_pose(self, timestep, sensor_id):
        if self.kaptures.trajectories is not None and (timestep, sensor_id) in self.kaptures.trajectories:
            pose_world_to_cam = self.kaptures.trajectories[(timestep, sensor_id)]
            pose_world_to_cam_matrix = np.zeros((4, 4), dtype=np.float)
            pose_world_to_cam_matrix[0:3, 0:3] = quaternion.as_rotation_matrix(pose_world_to_cam.r)
            pose_world_to_cam_matrix[0:3, 3] = pose_world_to_cam.t_raw
            pose_world_to_cam_matrix[3, 3] = 1.0
            T = torch.tensor(pose_world_to_cam_matrix).float()
            gt_pose=T.inverse() # gt_pose为从cam_to_world
        else:
            gt_pose=T=torch.eye(4)
        return gt_pose, pose_world_to_cam
    
    def __getitem__(self, index):
        image_id = self.dataset[index]
        timestep, sensor_id=self.sensor_dict[image_id]
        
        # camera intrinsics
        camera_params=np.array(self.kaptures.sensors[sensor_id].camera_params[2:])
        K = np.array([[camera_params[0],0,camera_params[1]],
                    [0,camera_params[0],camera_params[2]],
                    [0,0,1]])
        
        # T from point cloud to camera
        gt_pose, gt_pose_world_to_cam_q=self.load_pose(timestep, sensor_id) # camera to world
        gt_pose_world_to_cam_q = np.concatenate((gt_pose_world_to_cam_q.t_raw, gt_pose_world_to_cam_q.r_raw))
        T_c2w = gt_pose.numpy() # camera to world
        T_w2c = np.linalg.inv(T_c2w)
        
        # T from world to voxel coordinate 将坐标系移到camera附近，高度不动
        T_w2v = np.eye(4).astype(np.float32)
        T_w2v[:2,3] = -T_c2w[:2,3]
        T_w2v_inv = np.linalg.inv(T_w2v).copy()
        
        # ------------- load image, original size is 1080x1920 -------------
        img = cv2.imread(os.path.join(self.input_path, 'sensors/records_data', image_id))
        depth_map_path = os.path.join(self.input_path, 'sensors/depth_data', image_id.replace("image", "depth"))
        depth_map = Image.open(depth_map_path) # RGBA
        depth_map = np.array(depth_map)
        
        R = depth_map[:,:,0].astype(np.float32)
        G = depth_map[:,:,1].astype(np.float32)
        B = depth_map[:,:,2].astype(np.float32)
        normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1)
        depth_map = 1000 * normalized

        # origin image
        if self.opt.vis_debug:
            cv2.imwrite(f'z_dataset/img_ori_{index}.png', img)
        
        # scale to 360x640
        new_size = (int(round(img.shape[1] * self.opt.img_scale)), int(round((img.shape[0] * self.opt.img_scale))))
        img = cv2.resize(img,
                         new_size,
                         interpolation=cv2.INTER_LINEAR)
        depth_map_image = Image.fromarray(depth_map)
        resized_depth_map_nearest = depth_map_image.resize(new_size, Image.NEAREST)
        depth_map = np.array(resized_depth_map_nearest)
        K = self.camera_matrix_scaling(K, self.opt.img_scale)
        
        if 'train' == self.mode:
            img_crop_dx = random.randint(0, img.shape[1] - self.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - self.img_H)
        else:
            img_crop_dx = int((img.shape[1] - self.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.img_H) / 2)
        img = img[img_crop_dy:img_crop_dy + self.img_H,
              img_crop_dx:img_crop_dx + self.img_W, :]
        depth_map = depth_map[img_crop_dy:img_crop_dy + self.img_H,
              img_crop_dx:img_crop_dx + self.img_W]
        K = self.camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)
        
        # resize and cropped image
        if self.opt.vis_debug:
            cv2.imwrite(f'z_dataset/img_resize_crop_{index}.png', img)
        
        # ------------- load point cloud ----------------
        npy_data = self.voxel_points.copy() # important! keep self.voxel points unchanged
        npy_data = npy_data[:, np.random.permutation(npy_data.shape[1])]
        pc_np = npy_data[0:3, :]  # 3xN
        intensity_np = np.zeros((1, pc_np.shape[1]), dtype=np.float32)  # 1xN
        surface_normal_np = np.zeros((3, pc_np.shape[1]), dtype=np.float32)  # 3xN

        # origin pcd
        if self.opt.vis_debug:
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(pc_np.T)
            open3d.io.write_point_cloud(f'z_dataset/input_pcd_{index}.ply', debug_point_cloud)
        
        # transform frame to voxel center
        pc_homo_np = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)), axis=0)  # 4xN
        Pr_pc_homo_np = np.dot(T_w2v, pc_homo_np)  # 4xN
        pc_np = Pr_pc_homo_np[0:3, :]  # 3xN
        if self.opt.vis_debug:
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(pc_np.T)
            open3d.io.write_point_cloud(f'z_dataset/input_pcd_T_w2v_{index}.ply', debug_point_cloud)

        # limit max_z, the pc is in CAMERA coordinate
        pc_np_x_square = np.square(pc_np[0, :])
        pc_np_y_square = np.square(pc_np[1, :])
        pc_np_range_square = pc_np_x_square + pc_np_y_square
        pc_mask_range = pc_np_range_square < self.opt.pc_max_range * self.opt.pc_max_range
        pc_np = pc_np[:, pc_mask_range]
        intensity_np = intensity_np[:, pc_mask_range]
        surface_normal_np = surface_normal_np[:, pc_mask_range]
        if self.opt.vis_debug:
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(pc_np.T)
            open3d.io.write_point_cloud(f'z_dataset/input_pcd_T_w2v_limit_{index}.ply', debug_point_cloud)
        
        # point cloud too huge, voxel grid downsample first
        if pc_np.shape[1] > 4 * self.opt.num_pc:
            # point cloud too huge, voxel grid downsample first
            pc_np, intensity_np, surface_normal_np = self.downsample_with_intensity_sn(pc_np, intensity_np, surface_normal_np, voxel_grid_downsample_size=0.4)
            pc_np = pc_np.astype(np.float32)
            intensity_np = intensity_np.astype(np.float32)
            surface_normal_np = surface_normal_np.astype(np.float32)
            
        # random sampling
        pc_np, intensity_np, surface_normal_np = self.downsample_np(pc_np, intensity_np, surface_normal_np, self.opt.num_pc)
        
        #  ------------- apply random transform on points under the NWU coordinate ------------
        if 'train' == self.mode:
            Pr = self.generate_random_transform()
            Pr_inv = np.linalg.inv(Pr)

            # -------------- augmentation ----------------------
            # pc_np, intensity_np = self.augment_pc(pc_np, intensity_np)
            if random.random() > 0.5:
                img = self.augment_img(img)
        elif 'val_random_Ry' == self.mode:
            Pr = self.generate_random_transform(0, 0, 0,
                                                0, math.pi*2, 0)
            Pr_inv = np.linalg.inv(Pr)
        elif 'val' == self.mode or 'test' == self.mode:
            Pr = np.identity(4, dtype=np.float32)
            Pr_inv = np.identity(4, dtype=np.float32)

        
        P = T_w2c @ T_w2v_inv @ Pr_inv # 对于输入点云的新GT Pose
        
        # then aug to get final input pcd
        pc_homo_np = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)), axis=0)  # 4xN
        Pr_pc_homo_np = np.dot(Pr, pc_homo_np)  # 4xN
        pc_np = Pr_pc_homo_np[0:3, :]  # 3xN
        if self.opt.vis_debug:
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(pc_np.T)
            open3d.io.write_point_cloud(f'z_dataset/input_pcd_T_w2v_aug_{index}.ply', debug_point_cloud)
        
        # input pcd in cam coordinate frame
        pc_homo_np = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)), axis=0)  # 4xN
        Pr_pc_homo_np = np.dot(P, pc_homo_np)  # 4xN
        pc_np_in_cam = Pr_pc_homo_np[0:3, :]  # 3xN
        if self.opt.vis_debug:
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(pc_np_in_cam.T)
            open3d.io.write_point_cloud(f'z_dataset/input_pcd_in_cam_{index}.ply', debug_point_cloud)
        
        # 2. get multi-level points and neighbor indexes for pyramid feature map
        num_stages = 5
        data_dict = precompute_point_cloud_stack_mode(pc_np, intensity_np, surface_normal_np, lengths=self.num_pc, num_stages=5)
        feats = torch.from_numpy(np.concatenate([intensity_np, surface_normal_np], axis=0).T.astype(np.float32))  

        data_dict['feats'] = feats

        coarse_points = np.array(data_dict['points'][-1], dtype=np.float32).T  # [3, 1280]
        for i in range(num_stages):
            # data_dict['points'][i] = torch.from_numpy(np.asarray(data_dict['points'][i].points, dtype=np.float32))
            data_dict['neighbors'][i] = data_dict['neighbors'][i].long()
            if i < num_stages - 1:
                data_dict['subsampling'][i] = data_dict['subsampling'][i].long()
                data_dict['upsampling'][i] = data_dict['upsampling'][i].long()
        
        #get 1/8 scale image for correspondences
        scale_size = 0.125
        K_2 = self.camera_matrix_scaling(K,0.5) # 对应fine resolution (1/2) img的K
 
        K_4=self.camera_matrix_scaling(K,scale_size) # 对应coarse resolution (1/8) img的K
    
        # TODO: ORIGIN

        # # project coarse_points to image_s8 and get corrs
        # [3, 1280]
        proj_coarse_points = np.dot(K_4, np.dot(P[0:3, 0:3], coarse_points) + P[0:3, 3:]) # 1280个点投影到coarse image上
        coarse_points_mask = np.zeros((1, np.shape(coarse_points)[1]), dtype=np.float32) 
        proj_coarse_points[0:2, :] = proj_coarse_points[0:2, :] / proj_coarse_points[2:, :] # 1280个点投影到coarse image上的坐标
        xy = np.floor(proj_coarse_points[0:2, :] + 0.5)
        
        depth_map_image = Image.fromarray(depth_map)
        resized_depth_map_nearest = depth_map_image.resize((int(self.img_W*scale_size), int(self.img_H*scale_size)), Image.NEAREST)
        depth_map_s8 = np.array(resized_depth_map_nearest)
        min_depth = depth_map_s8[np.clip(xy[1,:], 0, int(self.img_H*scale_size - 1)).astype(np.int32), 
                                     np.clip(xy[0,:], 0, int(self.img_W*scale_size - 1)).astype(np.int32)]
        depth_thres = 3
        
        is_in_picture = (xy[0, :] >= 1) & (xy[0, :] <= (self.img_W*scale_size - 3)) \
                        & (xy[1, :] >= 1) & (xy[1, :] <= (self.img_H*scale_size - 3)) \
                        & (proj_coarse_points[2, :] > 0)
        is_in_picture_no_occ = (xy[0, :] >= 1) & (xy[0, :] <= (self.img_W*scale_size - 3)) \
                            & (xy[1, :] >= 1) & (xy[1, :] <= (self.img_H*scale_size - 3)) \
                            & (proj_coarse_points[2, :] > 0) & (proj_coarse_points[2, :] < min_depth + depth_thres)
        coarse_points_mask[:, is_in_picture] = 1.
        
        # vis debug
        if self.opt.vis_debug:
            # coarse points
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(coarse_points.T)
            open3d.io.write_point_cloud(f'z_dataset/coarse_points_{index}.ply', debug_point_cloud)
            
            # in_picture coarse points
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(coarse_points[:,is_in_picture].T)
            open3d.io.write_point_cloud(f'z_dataset/coarse_points_in_pic_{index}.ply', debug_point_cloud)
            
            # in_picture no occ coarse points
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(coarse_points[:,is_in_picture_no_occ].T)
            open3d.io.write_point_cloud(f'z_dataset/coarse_points_in_pic_no_occ_{index}.ply', debug_point_cloud)
            

        # sample 64 in image coarse points (no occ)
        pc_kpt_idx=np.where(is_in_picture_no_occ==1)[0] 
        if len(pc_kpt_idx) >= self.num_kpt:
            p_index=np.random.permutation(len(pc_kpt_idx))[0:self.num_kpt] # 选取64个可以投影到image内的3D point的idx下标
            pc_kpt_idx=pc_kpt_idx[p_index]
        else:
            print(f"in image pc no occ only {len(pc_kpt_idx)}")
            fix_idx = np.asarray(range(len(pc_kpt_idx)))
            while len(pc_kpt_idx) + fix_idx.shape[0] < self.num_kpt:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(len(pc_kpt_idx)))), axis=0)
            random_idx = np.random.choice(len(pc_kpt_idx), self.num_kpt - fix_idx.shape[0], replace=False)
            p_index = np.concatenate((fix_idx, random_idx), axis=0)
            pc_kpt_idx=pc_kpt_idx[p_index]

        # sample 64 out image coarse points
        pc_outline_idx=np.where(is_in_picture==0)[0]
        if len(pc_outline_idx) >= self.num_kpt:
            p_index=np.random.permutation(len(pc_outline_idx))[0:self.num_kpt] # 同理选取64个投影到image之外的3D point的idx下标
            pc_outline_idx=pc_outline_idx[p_index]
        else:
            print(f"in image pc no occ only {len(pc_outline_idx)}")
            if len(pc_outline_idx) == 0:
                pc_outline_idx = np.argsort(proj_coarse_points[2, :].reshape(-1))[-self.num_kpt:] # 如果所有coarse points都在image frustum内，则选取离相机最远的64个点作为pc_outline
            else:
                fix_idx = np.asarray(range(len(pc_outline_idx)))
                while len(pc_outline_idx) + fix_idx.shape[0] < self.num_kpt:
                    fix_idx = np.concatenate((fix_idx, np.asarray(range(len(pc_outline_idx)))), axis=0)
                random_idx = np.random.choice(len(pc_outline_idx), self.num_kpt - fix_idx.shape[0], replace=False)
                p_index = np.concatenate((fix_idx, random_idx), axis=0)
                pc_outline_idx=pc_outline_idx[p_index]
        
        if self.opt.vis_debug:
            # sampled good coarse points
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(coarse_points[:,pc_kpt_idx].T)
            open3d.io.write_point_cloud(f'z_dataset/coarse_points_sample_good_{index}.ply', debug_point_cloud)
            
            # sampled bad coarse points
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(coarse_points[:,pc_outline_idx].T)
            open3d.io.write_point_cloud(f'z_dataset/coarse_points_sample_bad_{index}.ply', debug_point_cloud)

        xy2 = xy[:, is_in_picture] # 3D Point投影到图片内的pixel坐标
        img_mask_s8 = coo_matrix((np.ones_like(xy2[0, :]), (xy2[1, :], xy2[0, :])), shape=(int(self.img_H*scale_size), int(self.img_W*scale_size))).toarray()
        img_mask_s8 = np.array(img_mask_s8)
        img_mask_s8[img_mask_s8 > 0] = 1. # 在coarse mask即1/8分辨率上 将被投影到的pixel位置为True
        
        if self.opt.vis_debug:
            cv2.imwrite(f'z_dataset/image_mask_s8_old_{index}.png', (img_mask_s8*255).astype(np.uint8))
            
        # use depth projection to get img_mask for img_outline_index
        resize_w = img_mask_s8.shape[1]
        resize_h = img_mask_s8.shape[0]
        x = np.arange(resize_w)
        y = np.arange(resize_h)
        xv, yv = np.meshgrid(x, y)

        keypoints = np.vstack([xv.ravel(), yv.ravel()]).T
        keypoints = keypoints.astype(np.int16)
        depths = depth_map_s8.reshape(-1)
        depth_mask = depths < 100
        # project into 3D points
        dense_point = self.projector.proj_2to3(keypoints, depths, K_4, T_w2v@T_c2w, depth_unit=1)
        if self.opt.vis_debug:
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(dense_point[depth_mask])
            open3d.io.write_point_cloud(f'z_dataset/dense_point_{index}.ply', debug_point_cloud)
        dense_point = dense_point.reshape(resize_h, resize_w, 3).astype(np.float32)
        img_dist = np.linalg.norm(dense_point, axis=2)
        img_mask_s8 = img_dist <= self.opt.pc_max_range # 所有投影到3D点后，在输入点云范围内的点
        
        if self.opt.vis_debug:
            cv2.imwrite(f'z_dataset/image_mask_s8_new_{index}.png', (img_mask_s8*255).astype(np.uint8))
            
        coarse_xy = xy[:, pc_kpt_idx] # 选取的64个点在coarse image上对应pixel坐标
        img_kpt_s8_index=xy[1,pc_kpt_idx]*self.img_W*scale_size +xy[0,pc_kpt_idx] # 64个点对应的pixel（实际上是patch）的flatten下标，即image中正样本下标
        
        img_outline_index=np.where(img_mask_s8.squeeze().reshape(-1)==0)[0]
        if len(img_outline_index) < self.num_kpt:
            img_outline_index=np.argsort(depth_map_s8.reshape(-1))[-self.num_kpt:] # 若在voxel之外的pixel数量少于64个，则选取离相机最远（深度最大）的64个pixel
        else:
            p_index=np.random.permutation(len(img_outline_index))[0:self.num_kpt]
            img_outline_index=img_outline_index[p_index] # 选取64个image中的负样本下标

        # project to 1/2 resolution image
        coarse_kpts = coarse_points[:, pc_kpt_idx] # 64个点的3D坐标
        proj_points = np.dot(K_2, np.dot(P[0:3, 0:3], coarse_kpts) + P[0:3, 3:]) # 64个点投影到fine image上
        proj_points[0:2, :] = proj_points[0:2, :] / proj_points[2:, :]
        fine_xy = np.floor(proj_points[0:2, :])
        fine_is_in_picture = (fine_xy[0, :] >= 0) & (fine_xy[0, :] <= (self.img_W*0.5 - 1)) & (fine_xy[1, :] >= 0) & (fine_xy[1, :] <= (self.img_H*0.5 - 1)) & (proj_points[2, :] > 0)

        assert np.all(fine_is_in_picture==True)

        # get coarse inline points on fine feature map 
        fine_xy_kpts_index = fine_xy[1,:]*self.img_W*0.5 +fine_xy[0,:] # 对应pixel在fine image中的flatten下标
        fine_center_kpts_coors = coarse_xy * 4 # 对应pixel在fine image中的pixel坐标
        indices = point2node(data_dict['points'][1], data_dict['points'][-1][pc_kpt_idx]) # 1层即10240个点与64个选中的3D点之间最近的点（实际上是自身），即64个选中点在10240个点中的idx下标

        return {'img': torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1).contiguous(),
                'pc_data_dict': data_dict,
                'fine_pc_inline_index': indices.long(),
                'K': torch.from_numpy(K_2.astype(np.float32)),
                'K_4': torch.from_numpy(K_4.astype(np.float32)),
                'P': torch.from_numpy(P.astype(np.float32)),

                # 'pc_mask': torch.from_numpy(pc_mask).float(),       #(1,20480)
                'coarse_img_mask': torch.from_numpy(img_mask_s8).float(),     #(40,128)
                # 'img_mask': torch.from_numpy(img_mask).float(),

                'pc_kpt_idx': torch.from_numpy(pc_kpt_idx),         #128
                'pc_outline_idx':torch.from_numpy(pc_outline_idx), 
                'fine_xy_coors':torch.from_numpy(fine_xy.astype(np.int32)), 
                'coarse_img_kpt_idx':torch.from_numpy(img_kpt_s8_index).long() ,
                'fine_img_kpt_index':torch.from_numpy(fine_xy_kpts_index).long() ,      #128
                'fine_center_kpt_coors':torch.from_numpy(fine_center_kpts_coors.astype(np.int32)),
                'coarse_img_outline_index':torch.from_numpy(img_outline_index).long(),

                }
               

if __name__ == '__main__':
    from data.options_carla import Options_CARLA
    opt = Options_CARLA()
    debug_dataset, debug_dataloader = make_carla_dataloader(mode="train", opt=opt)
    debug_dataset, debug_dataloader = make_carla_dataloader(mode="val", opt=opt)
    debug_dataset, debug_dataloader = make_carla_dataloader(mode="test", opt=opt)