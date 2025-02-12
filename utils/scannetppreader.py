import cv2
import json
from glob import glob
import numpy as np
import open3d as o3d
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from reconstruction import Reconstruction
from tqdm import tqdm

from torch.utils.data import Dataset


class ScanNetPPReader(Dataset):
    def __init__(self,root_dir,scene_name):
        self.scene = scene_name
        self.root_dir = root_dir
        self.parent_dir = root_dir+'/data/{}/iphone/'.format(scene_name)
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
    
if __name__ == '__main__':
    voxel_size = 0.01
    res = 4
    depth_scale = 1000.0
    depth_max = 5.0
    n_labels = 21

    root_dir = '/home/motion/extra_storage/scannet_pp' # where scannet ++ is stored
    scene = '1d003b07bd' # the scene you wish to load

    reader = ScanNetPPReader(root_dir,scene)
    # rec = Reconstruction(depth_scale = depth_scale,depth_max=depth_max,res = res,voxel_size = voxel_size,n_labels = None,integrate_color = True,
    #     device = o3d.core.Device('CUDA:0'),miu = 0.001,weight_threshold = 5)
    rec = Reconstruction(depth_scale = depth_scale,depth_max=depth_max,res = res,voxel_size = voxel_size,n_labels = None,integrate_color = True,
        device = o3d.core.Device('CUDA:0'),miu = 0.001)
    for idx,dct in tqdm(enumerate(reader)):
        depth = dct['depth']
        rgb = dct['color']
        pose = dct['pose']
        intrinsic = dct['intrinsics_depth']
        rec.update_vbg(depth,intrinsic,pose,rgb,None)
        if(idx>300):
            break

    
    # mesh,label = rec.extract_triangle_mesh(weight_thold = 20)
    mesh,label = rec.extract_triangle_mesh()
    o3d.visualization.draw_geometries([mesh])