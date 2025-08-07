import json
import os
import pickle
import sys
from copy import deepcopy
from glob import glob
import numpy as np
import open3d as o3d
import torch
from torchmetrics.functional import jaccard_index
from tqdm.notebook import tqdm
import gc

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
data_dir = os.path.join(parent_dir, 'data')

from utils.my_calibration import (
    BrierScore3D,
    mECE_Calibration_calc_3D,
    mECE_Calibration_calc_3D_fix,
)
from utils.ScanNet_scene_definitions import (
    get_classes,
    get_scannetpp_classes,
    get_larger_test_and_validation_scenes,
    get_filenames, get_small_test_scenes2,
    get_scannetpp_test_scenes

)

from utils.scannetpp_utils import (
    scanentpp_gt_getter
)
fnames = get_filenames()

results_dir = fnames['results_dir']


test_scenes = get_scannetpp_test_scenes()

selected_scenes = test_scenes
gt_getter = scanentpp_gt_getter(
    fnames['ScanNetpp_root_dir'],
    os.path.join(data_dir, 'scannetpp_class_equivalence_revised.xlsx')
)
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

scene_data = []
for scene in tqdm(selected_scenes):
    _, gt_labels = gt_getter.get_gt_point_cloud_and_labels(scene)
    print((gt_labels).shape)
    print((gt_labels).min())
    scene_row = np.zeros(151, dtype=int)
    
    unique_labels = np.unique(gt_labels)  
    for label in unique_labels:
        if 0 <= label <= 150: 
            scene_row[label] = 1
    
    scene_data.append(scene_row)

scene_names = [f"{selected_scenes[i]}" for i in range(len(selected_scenes))]
df = pd.DataFrame(scene_data, index=scene_names, columns=range(151))

output_file = os.path.join(data_dir, "scene_labels_grid.xlsx")
df.to_excel(output_file)
