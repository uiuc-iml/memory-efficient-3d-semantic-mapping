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

evaluation_start = 0
no_void =  False
fnames = get_filenames()

results_dir = fnames['results_dir']
results_dir = '/work/hdd/bebg/results/scannetpp'
num_classes = 101
def get_experiments_and_short_names():
    a = json.load(open('../settings/experiments_and_short_names.json','r'))
    experiments = a['experiments']
    short_names = a['short_names']
    return experiments,short_names

def compute_mIoUs():
    accuracies = []
    experiments,short_names = get_experiments_and_short_names()
    
    
    # val_scenes,test_scenes = get_larger_test_and_validation_scenes()
    # test_scenes = get_larger_test_and_validation_scenes()
    test_scenes = get_scannetpp_test_scenes()

    selected_scenes = test_scenes
    gt_getter = scanentpp_gt_getter(fnames['ScanNetpp_root_dir'], 'mapping_to_top_100.xlsx')
    pcds_template = '{}/{}/{}/*.pcd'
    labels_template = '{}/{}/{}/labels*.p'
    per_exp_IoUs = {}
    per_exp_absents = {}
    for experiment in experiments:
        IoUs = {}
        absents = {}
        totals_gt = []
        totals = []
        for scene in tqdm(selected_scenes,desc = '{} mIoUs'.format(experiment),position = 1):
            try:
                print(scene)
                # gt_pcd_file = '{}/reconstruction_gts/gt_pcd_{}.pcd'.format(results_dir,scene)
                # gt_labels_file = '{}/reconstruction_gts/gt_labels_{}.p'.format(results_dir,scene)

                # gt_pcd = o3d.io.read_point_cloud(gt_pcd_file)
                # gt_labels= pickle.load(open(gt_labels_file,'rb'))
                gt_pcd, gt_labels = gt_getter.get_gt_point_cloud_and_labels(scene)
                
                # print((pcds_template.format(results_dir,experiment,scene)))
                pcd_files = sorted(glob(pcds_template.format(results_dir,experiment,scene)))
                
                label_files = sorted(glob(labels_template.format(results_dir,experiment,scene)))
                
                pcd_file = pcd_files[-1]
                labels_file = label_files[-1]
                pcd = o3d.io.read_point_cloud(pcd_file)
                
                labels= pickle.load(open(labels_file,'rb'))
                # print(f' Shape of the ground truth labels{gt_labels.shape}')
                # print(f'shape of the recontructed pcd labels: {labels.shape}')
                

                # we then unscramble the labels:

                pcd_tree = o3d.geometry.KDTreeFlann(gt_pcd)
                points = np.asarray(pcd.points)
                unscrambler = np.zeros(points.shape[0]).astype(np.int64)
                scrambler = np.zeros(points.shape[0]).astype(np.int64)
                for i in range(points.shape[0]):
                    [k, idx, dist] = pcd_tree.search_knn_vector_3d(points[i],1)
                    # unscrambler[idx[0]] = i
                    scrambler[i] = idx[0]
                # diff_labels = labels[unscrambler]
                diff_labels = labels
                gt_labels = gt_labels[scrambler]
                print(gt_labels.shape)
                # print(labels.shape)

                diff_labels = np.argmax(diff_labels, axis=1)
                # gt_labels = np.argmax(gt_labels,axis =1)
                totals_gt.extend(gt_labels.tolist())
                totals.extend(diff_labels.tolist())
                unique_gt = np.unique(gt_labels).tolist()
                unique_pred = np.unique(diff_labels).tolist()
                absent = set(list(range(num_classes))) - set(unique_gt + unique_pred)
                IoU = jaccard_index(task = 'multiclass',preds = torch.from_numpy(diff_labels),target= torch.from_numpy(gt_labels),num_classes = num_classes, ignore_index=num_classes,average = None)
                IoUs.update({scene:IoU})
                absents.update({scene:absent})
                del pcd_tree
                del gt_labels
                del labels
                del diff_labels
                del points
                del pcd
                gc.collect()
            except Exception as e:
                print(scene,'mIoU Reconstruction',e)
                continue
        IoU = jaccard_index(task = 'multiclass',preds=torch.from_numpy(np.array(totals)),target = torch.from_numpy(np.array(totals_gt)),num_classes = num_classes,ignore_index=num_classes,average = None)
        non_null = np.array(totals_gt) != num_classes
        accuracy = (np.array(totals)[non_null] == np.array(totals_gt)[non_null]).sum()/np.array(totals)[non_null].shape[0]
        accuracies.append(accuracy)
        pred = np.array(totals)[non_null].tolist()     
        absent = set(list(range(num_classes))) - set(pred + totals_gt)
        IoUs.update({'aggregate':IoU})
        absents.update({'aggregate':absent})
        per_exp_IoUs.update({experiment:IoUs})
        per_exp_absents.update({experiment:absents})
        print('here')

    a = IoUs['aggregate'].numpy()
    a[a!=0].mean()

    dfs = []
    for experiment,short_name in zip(experiments,short_names):
        IoUs = per_exp_IoUs[experiment]
        absents = per_exp_absents[experiment]
        expanded_scenes = selected_scenes+['aggregate']
        metrics = np.zeros((len(expanded_scenes),num_classes))
        for i,scene in enumerate(expanded_scenes):
            print(scene)
            metrics[i,:] = IoUs[scene]
            if(absents[scene]):
                metrics[i,np.array(list(absents[scene]))] = np.nan

        import pandas as pd

        df = pd.DataFrame(metrics)
        classes = get_scannetpp_classes()
        df.columns = classes
        df.loc[:,'scene'] = expanded_scenes
        df.loc[:,'experiment'] = short_name
        dfs.append(df)
    # df.to_pickle('./Results/3D IoUs/Larger/{} 3D IoUs.p'.format(experiment))

    final_df = pd.concat(dfs).reset_index(drop = True)
    final_df.to_excel('{}/quant_eval/3D IoUs per experiment.xlsx'.format(results_dir))

    # final_df.to_excel('{}/Comparative Experiments 3D IoUs.xlsx'.format(results_dir))
    selected_df = final_df.loc[final_df.scene == 'aggregate',:].reset_index(drop = True)
    df = selected_df.transpose()
    df.rename(columns = df.loc['experiment',:],inplace = True)
    df.drop('experiment',inplace = True)
    df.drop('scene',inplace = True)
    # df.drop('irrelevant',inplace = True)
    df.loc['mIoU',:] = df.mean(axis = 0)
    df.loc['Accuracy'] = accuracies
    selected_df = df.reset_index(drop = False)
    transposed_df = selected_df.transpose()
    
    transposed_df.to_excel('{}/quant_eval/3D IoUs.xlsx'.format(results_dir))





def compute_mECEs():
    
    experiments,short_names = get_experiments_and_short_names()
    # classes = get_classes()
    classes = get_scannetpp_classes

    # cal_scenes,test_scenes = get_larger_test_and_validation_scenes()
    # selected_scenes = test_scenes
    # selected_scenes = get_larger_test_and_validation_scenes()
    test_scenes = get_scannetpp_test_scenes()

    selected_scenes = test_scenes
    gt_getter = scanentpp_gt_getter(fnames['ScanNetpp_root_dir'], 'mapping_to_top_100.xlsx')


    ECE_by_experiment = []

    for g,experiment in enumerate(experiments):
        mECE_cal = mECE_Calibration_calc_3D(no_void = no_void, one_hot=False, n_classes=num_classes)
        TL_ECE_cal = mECE_Calibration_calc_3D_fix(no_void = no_void, one_hot=False, n_classes=num_classes)
        pcds_template = '{}/{}/{}/*.pcd'
        labels_template = '{}/{}/{}/labels*.p'
        gts = []
        preds = []
        print('\n\n experiment = {} \n\n\n '.format(experiment))
        for scene in tqdm(selected_scenes,desc = 'ECEs',position = 2):
            try:

                # gt_pcd_file = '{}/reconstruction_gts/gt_pcd_{}.pcd'.format(results_dir,scene)
                # gt_labels_file = '{}/reconstruction_gts/gt_labels_{}.p'.format(results_dir,scene)
                # gt_pcd = o3d.io.read_point_cloud(gt_pcd_file)
                # gt_labels= pickle.load(open(gt_labels_file,'rb'))
                gt_pcd, gt_labels = gt_getter.get_gt_point_cloud_and_labels(scene)
                pcd_file = sorted(glob(pcds_template.format(results_dir,experiment,scene)))[-1]
                labels_file = sorted(glob(labels_template.format(results_dir,experiment,scene)))[-1]
                pcd = o3d.io.read_point_cloud(pcd_file)
                labels= pickle.load(open(labels_file,'rb')).astype(np.float64)

                if(np.any(labels<0)):
                    labels = labels - labels.max(axis = 1).reshape((-1,1))
                    exp_labels = np.exp(labels)
                    labels = exp_labels/exp_labels.sum(axis=1).reshape((-1,1))
                if(np.any(labels.sum(axis = 1) > 1)):
                    labels = labels/labels.sum(axis = 1,keepdims = True)
                labels = labels/labels.sum(axis = 1,keepdims = True)

                pcd_tree = o3d.geometry.KDTreeFlann(gt_pcd)
                points = np.asarray(pcd.points)
                scrambler = np.zeros(points.shape[0]).astype(np.int64)
                for i in range(points.shape[0]):
                    [k, idx, dist] = pcd_tree.search_knn_vector_3d(points[i],1)
                #     print(points[i]-gt_points[idx],i,idx)
                    scrambler[i] = idx[0]
                labels[labels.sum(axis =1)==0] = 1/101.0
                # this_stage_labels = deepcopy(gt_labels)
                # this_stage_labels[:] = 1/150.0
                # this_stage_labels[scrambler] = labels
                # map_label = this_stage_labels.argmax(axis = 1)
                # map_gt = gt_labels.argmax(axis = 1)
                # gts.extend(map_gt)
                # preds.extend(map_label)
                gt_labels = gt_labels[scrambler]

                # update all_bins
                # mECE_cal.update_bins(this_stage_labels,gt_labels)
                # TL_ECE_cal.update_bins(this_stage_labels,gt_labels)
                mECE_cal.update_bins(labels,gt_labels)
                TL_ECE_cal.update_bins(labels,gt_labels)

            except Exception as e:
                print(e)
                continue
        ECEs =  mECE_cal.get_ECEs()
        ECEs.append(TL_ECE_cal.get_TL_ECE())
        ECE_by_experiment.append(ECEs)
    
    import pandas as pd
    mtx = np.array(ECE_by_experiment)

    df = pd.DataFrame(mtx)
    # classes = ['null','wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture',
    #            'counter','desk','curtain','refrigerator','shower curtain','toilet','sink','bathtub','otherfurniture','aggregate','TL-ECE']
    # classes = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 
    #     'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 
    #     'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 
    #     'radiator', 'glass', 'clock', 'flag', 'aggregate', 'TL-ECE']
    classes = [
    "wall", "ceiling", "floor", "table", "door", "ceiling lamp", "cabinet", "blinds", "curtain", "chair",
    "storage cabinet", "office chair", "bookshelf", "whiteboard", "window", "box", "window frame", "monitor",
    "shelf", "doorframe", "pipe", "heater", "kitchen cabinet", "sofa", "windowsill", "bed", "shower wall",
    "trash can", "book", "plant", "blanket", "tv", "computer tower", "kitchen counter", "refrigerator", "jacket",
    "electrical duct", "sink", "bag", "picture", "pillow", "towel", "suitcase", "backpack", "crate", "keyboard",
    "rack", "toilet", "paper", "printer", "poster", "painting", "microwave", "board", "shoes", "socket", "bottle",
    "bucket", "cushion", "basket", "shoe rack", "telephone", "file folder", "cloth", "blind rail", "laptop",
    "plant pot", "exhaust fan", "cup", "coat hanger", "light switch", "speaker", "table lamp", "air vent",
    "clothes hanger", "kettle", "smoke detector", "container", "power strip", "slippers", "paper bag", "mouse",
    "cutting board", "toilet paper", "paper towel", "pot", "clock", "pan", "tap", "jar", "soap dispenser",
    "binder", "bowl", "tissue box", "whiteboard eraser", "toilet brush", "spray bottle", "headphones", "stapler",
    "marker","null", "aggregate", "TL-ECE"]

    df.columns = classes
    df.loc[:,'experiments'] = short_names
    df = df.loc[:,[df.columns[-1]]+df.columns[:-1].tolist()]
    df.loc[:,'mECE - things'] = df.loc[:,df.columns[5:-2]].mean(axis = 1)
    if(no_void):
        df.loc[:,'mECE -all'] = df.loc[:,df.columns[1:-3]].mean(axis = 1)
        df.loc[:,'mECE - stuff'] = df.loc[:,df.columns[2:4]].mean(axis = 1)
    else:
        df.loc[:,'mECE -all'] = df.loc[:,df.columns[1:-3]].mean(axis = 1)
        df.loc[:,'mECE - stuff'] = df.loc[:,df.columns[1:4]].mean(axis = 1)
    cols = list(df.columns)
    cols2 = ['aggregate','mECE - things', 'mECE -all', 'mECE - stuff','TL-ECE']
    cols[-5:] = cols2
    df = df.loc[:,cols]
    df.to_excel('{}/quant_eval/ECEs by class and experiment_alt_finetuned.xlsx'.format(results_dir),index = False)
    
    
    
    # import seaborn as sns
    # sns.set_theme()
    # for i,c in tqdm(enumerate(classes[:21])):
    #     fig,axis = plt.subplots(1,len(experiments))

    #     for experiment,short_name in tqdm(zip(experiments,short_names)):
    #         ax = axis[experiments.index(experiment)]
    #         cc = all_cals[experiment].cals[i]
    #         cal,conf,lims = cc.return_calibration_results()
    #         lims = lims-0.05
    #         cal = cc.correct_bin_members/cc.total_bin_members
    #         cal = np.nan_to_num(cal,0)
    # #         fig, ax = plt.subplots()
    #         fig.set_size_inches(len(experiment)*8,8)
    #         p1 = ax.bar(x= lims, height = cal,  width = 0.8*(lims[1]-lims[0]),color = 'b',alpha = 0.5)
    #         ax.bar(x= lims, height = conf, width = 0.8*(lims[1]-lims[0]),color = 'r',alpha = 0.2)
    #         ax.plot(np.arange(11)/10,np.arange(11)/10)
    #         membership = (cc.total_bin_members/cc.total_bin_members.sum()*100)
    #         membership = np.round(membership,decimals = 1)
    #         membership = ['(' +str(i) + '%)' for i in membership]

    #         # membership = eval(np.array_str(membership, precision=2, suppress_small=True))
    #         ax.bar_label(p1, labels = membership,label_type='edge')
    #         ax.set_title('{} - {} - ECE = {:.3f}'.format(short_name,c,cc.get_ECE()))
    #         ax.set_xlabel('Upper Confidence')
    #         ax.set_ylabel('Empirical Accuracy within bin (total pixel %)')
    #     plt.savefig('{}/mECE Analysis/plots/Segformer3/Rel_diagrams_by_experiment_{}.png'.format(results_dir,c),bbox_inches = 'tight')
    #     plt.show()




def compute_brier_scores():
    # cal_scenes,test_scenes = get_larger_test_and_validation_scenes()
    # selected_scenes = test_scenes
    # selected_scenes = get_larger_test_and_validation_scenes()
    test_scenes = get_scannetpp_test_scenes()

    selected_scenes = test_scenes
    gt_getter = scanentpp_gt_getter(fnames['ScanNetpp_root_dir'], 'mapping_to_top_100.xlsx')

    experiments,short_names = get_experiments_and_short_names()
    metric = BrierScore3D
    ECE_by_experiment = []
    all_cals = {}
    per_scene_mECEs = []
    for g,experiment in enumerate(experiments):
        multiply = False
        mECE_cal = metric(no_void = True, n_classes=num_classes, one_hot=False)

        # cc_3d = Calibration_calc_3D(no_void = True)
        pcds_template = '{}/{}/{}/*.pcd'
        labels_template = '{}/{}/{}/labels*.p'
        splt = experiment.split('|')
        difs = []
        gts = []
        preds = []
        print('\n experiment = {} \n '.format(experiment))
        per_scene_mECE = []
        for scene in selected_scenes:
            try:
                per_scene_cal =  metric(no_void = True, n_classes=num_classes, one_hot=False)

                # gt_pcd_file = '{}/reconstruction_gts/gt_pcd_{}.pcd'.format(results_dir,scene)
                # gt_labels_file = '{}/reconstruction_gts/gt_labels_{}.p'.format(results_dir,scene)
                # gt_pcd = o3d.io.read_point_cloud(gt_pcd_file)
                # gt_labels= pickle.load(open(gt_labels_file,'rb'))
                gt_pcd, gt_labels = gt_getter.get_gt_point_cloud_and_labels(scene)
                pcd_file = sorted(glob(pcds_template.format(results_dir,experiment,scene)))[-1]
                labels_file = sorted(glob(labels_template.format(results_dir,experiment,scene)))[-1]
            #     for pcd_file,labels_file in zip(pcd_files,label_files):
                pcd = o3d.io.read_point_cloud(pcd_file)
                labels= pickle.load(open(labels_file,'rb'))
        #         print(np.unique(labels))
            #     label_logits = labels
                if(np.any(labels<0)):
                    labels = labels - labels.max(axis = 1).reshape((-1,1))
                    exp_labels = np.exp(labels)
                    labels = exp_labels/exp_labels.sum(axis=1).reshape((-1,1))
                if(np.any(labels > 1)):
                    labels = labels/labels.sum(axis = 1,keepdims = True)
                labels = labels/labels.sum(axis = 1,keepdims = True)
                labels[np.isnan(labels)] = 1/num_classes
                pcd_tree = o3d.geometry.KDTreeFlann(gt_pcd)
                points = np.asarray(pcd.points)
                gt_points = np.asarray(gt_pcd.points)
                unscrambler = np.zeros(points.shape[0]).astype(np.int64)
                scrambler = np.zeros(points.shape[0]).astype(np.int64)
                for i in range(points.shape[0]):
                    [k, idx, dist] = pcd_tree.search_knn_vector_3d(points[i],1)
                    scrambler[i] = idx[0]
                labels[labels.sum(axis =1)==0] = 1/101.0
                # this_stage_labels = deepcopy(gt_labels)
                # this_stage_labels[:] = 1/21.0
                # this_stage_labels[scrambler] = labels
                # map_label = this_stage_labels.argmax(axis = 1)
                # map_gt = gt_labels.argmax(axis = 1)
                gt_labels = gt_labels[scrambler]
                ma = np.argmax(gt_labels)
                # print(gt_labels[ma])

                # gts.extend(map_gt)
                # preds.extend(map_label)
                # mECE_cal.update_bins(this_stage_labels,map_gt)
                # per_scene_cal.update_bins(this_stage_labels,map_gt)
                # print('here1')
                mECE_cal.update_bins(labels,gt_labels)
                # print('here2')
                per_scene_cal.update_bins(labels,gt_labels)
                # print('here3')
                per_scene_mECE.append(per_scene_cal.return_score())
            except Exception as e:
                print(e)
                continue
        per_scene_mECEs.append(per_scene_mECE)
        all_cals.update({experiments[g]:mECE_cal})
        ECEs = mECE_cal.return_score()
        ECE_by_experiment.append(ECEs)

    import pandas as pd
    dfs = []
    mtx = np.array(ECE_by_experiment)
    df = pd.DataFrame(mtx)
    df.columns = ['Brier']
    df.loc[:,'experiments'] = short_names
    df = df.loc[:,[df.columns[-1]]+df.columns[:-1].tolist()]
    df = df.transpose()
    df.columns = df.loc['experiments',:]
    df = df.drop('experiments')
    df = df.reset_index(drop = False)
    if(not os.path.exists('{}/quant_eval'.format(results_dir))):
        os.makedirs('{}/quant_eval'.format(results_dir),exist_ok=True)
    df.to_excel('{}/quant_eval/brier_scores.xlsx'.format(results_dir),index = False)


from multiprocessing import Process

# p1 = Process(target = compute_mIoUs)
# p2 = Process(target = compute_mECEs)
# p3 = Process(target = compute_brier_scores)
# compute_mIoUs()
# compute_brier_scores()
# p1.start()
# p2.start()
# p3.start()
# p1.join()
# p2.join()
# p3.join()
compute_mIoUs()
compute_mECEs()
compute_brier_scores()

