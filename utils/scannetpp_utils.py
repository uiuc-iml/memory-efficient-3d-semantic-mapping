import pandas as pd
import numpy as np
import open3d as o3d
import json

class scanentpp_gt_getter:
    def __init__(self,root_dir,class_equivalence_dir):
        self.root_dir = root_dir +'/data'
        self.class_equivalence_df = equivalence_df = pd.read_excel(class_equivalence_dir).iloc[:,:4]
    def get_gt_point_cloud_and_labels(self,scene_name):
        scene_dir = '{}/{}'.format(self.root_dir,scene_name)
        semantics_path = scene_dir+'/scans/mesh_aligned_0.05_semantic.ply'
        segments_ano = scene_dir + '/scans/segments_anno.json'
        segments = scene_dir + '/scans/segments.json'        
        with open(segments_ano,'r') as f:
            segments_ano_dict = json.load(f)
        with open(segments,'r') as f:
            segments_dict = json.load(f)        
        
        mesh = o3d.io.read_triangle_mesh(semantics_path)
        
        tmp = np.array(segments_dict['segIndices'])
        (np.argsort(tmp)-np.arange(tmp.shape[0])).sum()
        mesh_df = pd.DataFrame({'idx':tmp,'class':tmp.shape[0]*[None]})
        segments_ano_dict['segGroups']
        for dct in segments_ano_dict['segGroups']:
            mesh_df.loc[dct['segments'],'class'] = dct['label']
        merge = pd.merge(mesh_df,self.class_equivalence_df,how = 'left',left_on = 'class',right_on = 'Scannet++ class')
        points = np.array(mesh.vertices)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        classes = merge.loc[:,'Segformer class Index'].values
        classes = np.nan_to_num(classes,100).astype(int)
        # classes = classes.reshape((classes.shape[0], 1))
        return pcd,classes
    
if __name__=='__main__':
    
    root_dir = '/home/motion/extra_storage/scannet_pp'
    class_equivalence_dir = 'scannetpp_class_equivalence_revised.xlsx'
    gt_getter = scanentpp_gt_getter(root_dir = root_dir,class_equivalence_dir = class_equivalence_dir)

    pcd,gt_labels = gt_getter.get_gt_point_cloud_and_labels('0a5c013435')
    print(np.asarray(pcd.points).shape,gt_labels.shape)
    print(gt_labels[400000:400200])