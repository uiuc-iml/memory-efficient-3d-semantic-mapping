# for param in model.parameters():
#     param.requires_grad = False
# decoder_head = model.decode_head
# decoder_head.classifier = nn.Conv2d(768,21,kernel_size=(1, 1), stride=(1, 1))
# for param in decoder_head.parameters():
#     param.requires_grad = True

from glob import glob
import pandas as pd 
import cv2
from joblib import Parallel,delayed
from datasets import Dataset
import numpy as np
import pickle
from datasets import IterableDataset
from tqdm import tqdm

# frames = pd.Series(sorted(glob('/home/motion/data/scannet_pp/data/**/gt_semantics/*.png',recursive = True)))
# frame_names = frames.str.split('/',expand = True).iloc[:,-1].str.split('.',expand = True).iloc[:,0]
# rgb_frames = frames.str.rsplit('/',expand = True,n = 2).iloc[:,0] + '/rgb/'+frame_names+'.jpg'

# img1 = '/home/motion/data/scannet_pp/data/1a8e0d78c0/iphone/gt_semantics/frame_000030.png'
# tmp = cv2.imread(frames[0],cv2.IMREAD_UNCHANGED)
# tmp2 = cv2.imread(rgb_frames[0])
# plt.figure(1)
# plt.imshow(tmp)
# plt.figure(2)
# plt.imshow(tmp2)
# plt.show()



original_train =  pd.read_csv('./nvs_sem_train.txt',header = None).iloc[:,0]
val_scenes = pd.read_csv('./newval.txt',header = None).iloc[:,0]
test_scenes =  pd.read_csv('./newtest.txt',header = None).iloc[:,0]
train_scenes = original_train.loc[np.logical_and(np.logical_not(original_train.isin(val_scenes)),np.logical_not(original_train.isin(test_scenes)))]
print(train_scenes.shape[0],original_train.shape[0])
# def data_generator():
i = 0
sfs = []
rgbs = []
all_frames = pd.Series(sorted(glob('/work/hdd/bebg/data/scannet_pp/data/**/gt_semantics/*.png',recursive = True)))

scene_names = all_frames.str.split('/',expand = True).iloc[:,-4]
train_semantic_images = all_frames.loc[scene_names.isin(train_scenes.iloc[:])]
val_semantic_images = all_frames.loc[scene_names.isin(val_scenes.iloc[:])]


def verify_all_data_and_create_dataset(semantic_frames):
    rgbs = []
    sfs = []
    frame_names = semantic_frames.str.split('/',expand = True).iloc[:,-1].str.split('.',expand = True).iloc[:,0]
    rgb_frames = semantic_frames.str.rsplit('/',expand = True,n = 2).iloc[:,0] + '/rgb/'+frame_names+'.jpg'
    n = rgb_frames.shape[0]
#     for semantic_frame,rgb_frame in tqdm(zip(semantic_frames.values,rgb_frames.values)):
#         try:
#             tmp = cv2.imread(semantic_frame,cv2.IMREAD_UNCHANGED)
#             tmp2 = cv2.imread(rgb_frame)
#             if((tmp is not None) and (tmp2 is not None)):
#                 rgbs.append(rgb_frame)
#                 sfs.append(semantic_frame)
#         except Exception as e:
#             i+=1
#             print(e)
#             continue
    all_verified_frames = Parallel(n_jobs = 6,verbose = 9,backend = 'threading')(delayed(verify_this_frame)(image_dirs)for image_dirs in zip(semantic_frames.values,rgb_frames.values))
    all_verified_frames = np.array(all_verified_frames)
    invalid = all_verified_frames =='none'
    invalid_entries = np.any(invalid,axis = 1)
    valid_entries = all_verified_frames[np.logical_not(invalid_entries)]
    rgbs = valid_entries[:,0]
    sfs = valid_entries[:,1]
    
    return {'pixel_values':rgbs,'label':sfs}

def verify_this_frame(image_dirs):
    semantic_frame,rgb_frame = image_dirs
    try:
        tmp = cv2.imread(semantic_frame,cv2.IMREAD_UNCHANGED)
        tmp2 = cv2.imread(rgb_frame)
        if((tmp is not None) and (tmp2 is not None)):
            return [rgb_frame,semantic_frame]
        else:
            return ['none','none']
        
    except Exception as e:
        i+=1
        print(e)
        return ['none','none']
        
train_dict = verify_all_data_and_create_dataset(train_semantic_images)
pickle.dump(train_dict,open('./scannet_pp_train_dict.p','wb'))
train_ds = Dataset.from_dict(train_dict)
train_ds.save_to_disk('./scannet_pp_finetune_train.hf')

val_dict = verify_all_data_and_create_dataset(val_semantic_images)
pickle.dump(val_dict,open('./scannet_pp_val_dict.p','wb'))
val_ds = Dataset.from_dict(val_dict)
val_ds.save_to_disk('./scannet_pp_finetune_val.hf')