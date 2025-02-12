# import yaml
# import klampt
# from klampt.math import se3
# from klampt.math import so3
# import cv2
import os
import sys
import nvidia_smi
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
# from reconstruction import Reconstruction
# from torch.utils.data import Dataset
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import open3d as o3d

# class BS3D_reader(Dataset):
#     def __init__(self,root_dir):
#         self.depth_dir = root_dir + '/depth_render/'
#         self.color_dir = root_dir + '/color/'
#         self.poses_dir = root_dir + '/poses.txt'
#         self.calibration_dir = root_dir + '/calibration.yaml'
#         with open(self.calibration_dir,'r') as stream:
#             tmp = yaml.safe_load(stream)
#             if tmp:
#                 print(tmp)
#             else:
#                 print("raise")
            
#         self.intrinsic = np.array(tmp['camera_matrix']['data']).reshape(3,3)
#         self.poses = pd.read_csv(self.poses_dir,sep = ' ',header = None)

#     def __len__(self):
#         return self.poses.shape[0]
        
#     def __getitem__(self,idx):
#         time_stamp = self.poses.iloc[idx,0]
#         t = self.poses.iloc[idx,1:4].values
#         q = self.poses.iloc[idx,4:].values
#         q[[0,1,2,3]] = q[[3,0,1,2]]
#         R = so3.ndarray(so3.from_quaternion(q))
#         pose = np.eye(4)
#         pose[:3,:3] = R
#         pose[:3,3] = t
#         depth_file = self.depth_dir + '{:.6f}.png'.format(time_stamp)
#         color_file = self.color_dir + '{:.6f}.jpg'.format(time_stamp)

#         depth = cv2.imread(depth_file,cv2.IMREAD_UNCHANGED)
#         rgb = cv2.imread(color_file,cv2.IMREAD_UNCHANGED)
#         rgb = cv2.resize(rgb,(0,0),fx = 1/2,fy = 1/2)

#         return {
#             'color': rgb,
#             'depth': depth,
#             'pose': pose,
#             'intrinsics_depth':self.intrinsic
#         }



# if __name__=='__main__':
#     root_dir = '/home/motion/bs3d/bs3d/cafeteria/cafeteria_depth_cam/depth_cam'


#     reader = BS3D_reader(root_dir)
#     voxel_size = 0.1
#     res = 4
#     depth_scale = 1000.0
#     depth_max = 5.0
#     n_labels = 150
#     rec = Reconstruction(depth_scale = depth_scale,depth_max=depth_max,res = res,voxel_size = voxel_size,n_labels = None,integrate_color = True,
#         device = o3d.core.Device('CUDA:0'),miu = 0.001)

#     for dct in tqdm(reader):
#         depth = dct['depth']
#         rgb = dct['color']
#         pose = dct['pose']
#         intrinsic = dct['intrinsics_depth']
#         rec.update_vbg(depth,intrinsic,pose,rgb,None)

#     mesh,label = rec.extract_point_cloud(return_raw_logits = False)
#     o3d.visualization.draw_geometries([mesh])

#     o3d.io.write_triangle_mesh('bsd_test.ply',mesh)
import yaml
import klampt
from klampt.math import se3
from klampt.math import so3
import cv2
from reconstruction import Reconstruction
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import open3d as o3d
from glob import glob
import json

def get_gpu_memory_usage():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return (info.used/(1024 ** 3))


class BS3D_reader(Dataset):
    def __init__(self,root_dir):
        self.depth_dir = root_dir + '/campus_depth_cam_depth_render/depth_render/'
        self.color_dir = root_dir + '/campus_depth_cam_color/color/'
        self.poses_dir = root_dir + '/campus_depth_cam_poses/poses.txt'
        self.calibration_dir = root_dir + '/campus_depth_cam_calibration/calibration.yaml'
        with open(self.calibration_dir,'r') as stream:
            tmp = yaml.safe_load(stream)
            if tmp:
                print(tmp)
            else:
                print("raise")
        # with open(self.calibration_dir,'r') as stream:
        #     tmp = yaml.safe_load(stream)
        self.intrinsic = np.array(tmp['camera_matrix']['data']).reshape(3,3)
        self.poses = pd.read_csv(self.poses_dir,sep = ' ',header = None)

    def __len__(self):
        return self.poses.shape[0]
        
    def __getitem__(self,idx):
        time_stamp = self.poses.iloc[idx,0]
        t = self.poses.iloc[idx,1:4].values
        q = self.poses.iloc[idx,4:].values
        q[[0,1,2,3]] = q[[3,0,1,2]]
        R = so3.ndarray(so3.from_quaternion(q))
        pose = np.eye(4)
        pose[:3,:3] = R
        pose[:3,3] = t
        depth_file = self.depth_dir + '{:.6f}.png'.format(time_stamp)
        color_file = self.color_dir + '{:.6f}.jpg'.format(time_stamp)

        depth = cv2.imread(depth_file,cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(color_file,cv2.IMREAD_UNCHANGED)
        rgb = cv2.resize(rgb,(0,0),fx = 1/2,fy = 1/2)

        return {
            'color': rgb,
            'depth': depth,
            'pose': pose,
            'intrinsics_depth':self.intrinsic
        }

class ScanNetPPReader(Dataset):
    def __init__(self,root_dir,scene_name):
        self.scene = scene_name
        self.root_dir = root_dir
        self.parent_dir = root_dir+'/data/{}/iphone'.format(scene_name)
        self.depth_image_files = sorted(glob(self.parent_dir+'/depth/*.png'))
        self.rgb_image_files =  sorted(glob(self.parent_dir+'/rgb/*.jpg'))
        with open(self.parent_dir + '/pose_intrinsic_imu.json','r') as f:
            self.poses_dict = json.load(f)
        self.poses_key = list(self.poses_dict.keys())
        self.size = len(self.depth_image_files)
        
    def __getitem__(self,key):
        depth_dir = self.depth_image_files[key]
        rgb_dir = self.rgb_image_files[key]
        pose_key = self.poses_key[key]
        depth = cv2.imread(depth_dir,cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(rgb_dir,cv2.IMREAD_UNCHANGED)
        rgb = cv2.resize(rgb,(0,0),fx = 1/7.5,fy = 1/7.5)
        aligned_pose = np.array(self.poses_dict[pose_key]['aligned_pose'])
        intrinsic = np.array(self.poses_dict[pose_key]['intrinsic'])
        intrinsic[:2,:3] = intrinsic[:2,:3]/7.5
        return {
            'color': rgb,
            'depth': depth,
            'pose': aligned_pose,
            'intrinsics_depth':intrinsic
        }

    def __len__(self):
        return self.size



if __name__=='__main__':
    root_dir = '/home/motion/extra_storage/scannet_pp'

    reader = ScanNetPPReader(root_dir, '7e09430da7')
    voxel_size = 0.025
    res = 8
    depth_scale = 1000.0
    depth_max = 5.0
    n_labels = 21
    # self.voxel_size = 0.025 #3.0 / 512
    # self.trunc =self.voxel_size * 8
    # self.res = 8
    # self.n_labels = n_labels
    # self.depth_scale = 1000.0
    # self.depth_max = 5.0
    # self.miu = 0.001
    arr_des = '/home/motion/semanticmapping/visuals/arrays/7e09430da7/cacherelease'
    # # plot_dir = os.path.join(des, 'topk')
    arr_dir = os.path.join(arr_des, f'reconstruction')
    # arr_dir = os.path.join(arr_des, f'scannetpp_Segformer_150_topk1')
    # # if not os.path.exists(plot_dir):
    # #     os.makedirs(plot_dir)
    if not os.path.exists(arr_dir):
        os.makedirs(arr_dir)
    block_count = []
    total_blocks = []
    hashmap_size = []
    gpu_memory_usage = []

    rec = Reconstruction(depth_scale = depth_scale,depth_max=depth_max,res = res,voxel_size = voxel_size,n_labels = None,integrate_color = False,
        device = o3d.core.Device('CUDA:0'),miu = 0.001)

    for dct in tqdm(reader):
        depth = dct['depth']
        # rgb = dct['color']
        pose = dct['pose']
        intrinsic = dct['intrinsics_depth']
        rec.update_vbg(depth,intrinsic,pose,None)
        gpu_memory_usage.append(get_gpu_memory_usage())
        gpu_memory_usage_np = np.array(gpu_memory_usage)
        np.save(os.path.join(arr_dir, "gpu_memory_usage.npy"), gpu_memory_usage_np)

    cpu_vbg = rec.vbg.cpu()
    mesh,label = rec.extract_point_cloud(return_raw_logits = False)
    o3d.visualization.draw_geometries([mesh])
    print(mesh)

    o3d.io.write_triangle_mesh('bsd_test.ply',mesh)