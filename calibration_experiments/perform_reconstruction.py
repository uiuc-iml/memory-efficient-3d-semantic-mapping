# from agents.utils.semantic_prediction import SemanticPredMaskRCNN
import argparse
import faulthandler
import gc
import json
import multiprocessing
import os
import pickle
import sys
import traceback
from functools import partial
# from tkinter import E
import psutil
import time
import torch
import nvidia_smi

import cv2

import os
import numpy as np
import open3d as o3d
import open3d.core as o3c
from klampt.math import se3
from tqdm import tqdm
from matplotlib import pyplot as plt

faulthandler.enable()

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

os.environ["OMP_NUM_THREADS"] = "2" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "2" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "2" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "2" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "2" # export NUMEXPR_NUM_THREADS=6


from experiment_setup import Experiment_Generator
from utils.ScanNet_scene_definitions import get_filenames, get_larger_test_and_validation_scenes, get_smaller_test_scenes, get_small_test_scenes2, get_fixed_train_and_val_splits, get_scannetpp_test_scenes
from utils.sens_reader import scannet_scene_reader, ScanNetPPReader, BS3D_reader

processes = 2



def get_gpu_memory_usage():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return (info.used/(1024 ** 3))

def save_plot(data, ylabel, title, filename):
    plt.figure()
    plt.plot(range(len(data)), data)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def reconstruct_scene(scene,experiment_name,experiment_settings,debug,oracle, n_labels = 101,dataset = "scannet++"):
    reconstruction_times = []
    gpu_memory_usage = []
    fnames = get_filenames()
    savedir = "{}/{}/".format(fnames['results_dir'], experiment_name)
    arr_dir = "{}/{}/visuals/".format(fnames['results_dir'], experiment_name)
    technique = experiment_settings['integration']
    if(technique == "topk" or technique == "Encoded Averaging" or technique == "MisraGries" or technique == "topk KH"):
        k = experiment_settings['k']
        arr_dir = "{}/{}/visuals/".format(fnames['results_dir'], f'{experiment_name}{k}')
        savedir = "{}/{}/".format(fnames['results_dir'], f'{experiment_name}{k}')

    if not os.path.exists(arr_dir):
        os.makedirs(arr_dir)


    EG = Experiment_Generator(n_labels=n_labels)
    
    rec,model = EG.get_reconstruction_and_model(experiment = experiment_settings,process_id = multiprocessing.current_process()._identity[0])
    if(experiment_settings['segmentation'] == "CLIP"):
        get_semantics = model.get_pixel_features
    else:
        if(technique == 'Generalized'):
            get_semantics = model.get_raw_logits
        # elif(technique == 'Histogram'):
        #     get_semantics = model.classify
        else:
            get_semantics = model.get_pred_probs
    if(dataset == "scannet++"):
        root_dir = fnames['ScanNetpp_root_dir']
    elif(dataset == "scannet"):
        root_dir = fnames['ScanNet_root_dir']
    elif(dataset == "bs3d"):
        root_dir = fnames['bs3d_root_dir']
    if(not os.path.exists(savedir)):
        try:
            os.mkdir(savedir)
        except Exception as e:
            print(e)
    if debug:
        lim = -1
    else:
        lim = -1
    folder = '{}/{}'.format(savedir,scene)
    arr_dir = '{}/{}'.format(arr_dir,scene)
    if not os.path.exists(folder):
        try:
            os.mkdir(folder)
        except Exception as e:
            print(e)
    try:
        device = o3d.core.Device('CUDA:0')

        if(dataset == "scannet"):
            my_ds = scannet_scene_reader(root_dir, scene ,lim = lim,disable_tqdm = True)
        elif(dataset == "scannet++"):
             my_ds = ScanNetPPReader(root_dir, scene)
        elif(dataset == "bs3d"):
            my_ds = BS3D_reader(root_dir)
        total_len = len(my_ds)

        if(lim == -1):
            lim = total_len
        randomized_indices = np.array(list(range(lim)))
        np.random.seed(0)
        proc_num = multiprocessing.current_process()._identity[0]%(processes+1) + 1
        print(f"total_len: {total_len}")
        for idx,i in tqdm(enumerate(randomized_indices),total = lim,desc = 'proc {}'.format(proc_num),position = proc_num):
            # start_time = time.time()
            # print("here2")
            try:
                data_dict = my_ds[i]
            except:
                print('\nerror while loading frame {} of scene {}\n'.format(i,scene))
                traceback.print_exc()
                continue
                
            depth = data_dict['depth']
            intrinsic = o3c.Tensor(data_dict['intrinsics_depth'][:3,:3].astype(np.float64))
            depth = o3d.t.geometry.Image(depth).to(device)
            try:
                color = data_dict['color']
                if(not isinstance(color,np.ndarray)):
                    continue
            except Exception as e:
                continue
            
            semantic_label = get_semantics(data_dict['color'],depth = data_dict['depth'],x = depth.rows,y = depth.columns)
            
            reconstruction_start_time = time.time()

            if(oracle):
                semantic_label_gt = cv2.resize(data_dict['semantic_label'],(depth.columns,depth.rows),interpolation= cv2.INTER_NEAREST)
                rec.update_vbg(data_dict['depth'],data_dict['intrinsics_depth'][:3,:3].astype(np.float64),
                            data_dict['pose'],semantic_label = semantic_label,semantic_label_gt = semantic_label_gt)
            else:
                rec.update_vbg(data_dict['depth'],data_dict['intrinsics_depth'][:3,:3].astype(np.float64),
                            data_dict['pose'],semantic_label = semantic_label, scene = scene)
            gpu_memory_usage.append(get_gpu_memory_usage())
            reconstruction_end_time = time.time()
            reconstruction_times.append(reconstruction_end_time - reconstruction_start_time)
            
            del intrinsic
            del depth

        save_plot(gpu_memory_usage, 'Memory Usage (GB)', 'GPU Memory Usage Over Iterations', os.path.join(arr_dir, 'gpu_memory_usage.png'))
        # Save reconstruction times plot
        save_plot(reconstruction_times, 'Time (s)', 'Reconstruction Time Per Iteration', os.path.join(arr_dir, 'reconstruction_times.png'))

        pcd,labels = rec.extract_point_cloud(return_raw_logits = False)
        o3d.io.write_point_cloud(folder+'/pcd_{:05d}.pcd'.format(idx), pcd, write_ascii=False, compressed=True, print_progress=False)
        pickle.dump(labels,open(folder+'/labels_{:05d}.p'.format(idx),'wb'))

        del rec
        gc.collect()

    except Exception as e:
        traceback.print_exc()
        del rec

def get_experiments():
    a = json.load(open('../settings/experiments_and_short_names.json','r'))
    experiments = a['experiments']
    return experiments


def main():
    import torch
    

    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", type=str, help="Integration method")
    parser.add_argument("--segmentation", type=str, help="Segmentation method")
    parser.add_argument("--k", type=int, help="k value for CTKH/EF")
    parser.add_argument("--num_labels", type=int, default=101, help="Number of labels")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset")
    parser.add_argument("--scene", type=str, help="Scene")

    parser.add_argument('--debug', action='store_true')
    parser.set_defaults(debug = False)
    parser.add_argument('--start', type=int, default=0,
                        help="""starting Reconstruction""")
    parser.add_argument('--end', type=int, default=-1,
                        help="""starting Reconstruction""")
    args = parser.parse_args()
        
    experiments = get_experiments()

    if(args.end == -1):
        experiments_to_do = experiments[args.start:]
    else:
        experiments_to_do = experiments[args.start:args.end]
    if(args.integration is not None and args.segmentation is not None):
        experiment_name = f"{args.segmentation} {args.integration}"
        experiments_to_do = [experiment_name]
        

    print('\n\n reconstructing {}\n\n'.format(experiments_to_do))
    for experiment in experiments_to_do:
        print(experiment)
        experiment_name = experiment
        experiment_settings = json.load(open('../settings/reconstruction_experiment_settings/{}.json'.format(experiment),'rb'))

        if(args.integration is not None and args.segmentation is not None):
            experiment_settings = {
            "integration": args.integration,
            "segmentation": args.segmentation,
            "k": args.k,
            "calibration": "None",
            "oracle": False,
            "L": 0,
            "epsilon": 1
            }
            num_labels = args.num_labels

        experiment_settings.update({'experiment_name':experiment_name})
        import multiprocessing
        debug = args.debug
        oracle = experiment_settings['oracle']
        if(args.dataset == "scannet"):
            val_scenes,test_scenes = get_larger_test_and_validation_scenes()
            num_labels = 21
        elif(args.dataset == "scannet++"):
            test_scenes = get_scannetpp_test_scenes()
            num_labels = 101
        elif(args.dataset == "bs3d"):
            test_scenes = ["campus"]
            num_labels = 150
        selected_scenes = test_scenes
        if(args.scene is not None):
            selected_scenes = [args.scene]
        p = multiprocessing.get_context('forkserver').Pool(processes = processes,maxtasksperchild = 1)

        res = []
        for a in tqdm(p.imap_unordered(partial(reconstruct_scene,experiment_name = experiment_name,experiment_settings=experiment_settings,debug = debug,oracle = oracle, n_labels=num_labels,dataset = args.dataset),selected_scenes,chunksize = 1), total= len(selected_scenes),position = 0,desc = 'tot_scenes'):
                res.append(a)        

        torch.cuda.empty_cache()
        o3d.core.cuda.release_cache()
    


if __name__=='__main__':
    main()