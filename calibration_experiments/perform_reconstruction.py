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
# import nvidia_smi

import cv2

# import os
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
from utils.sens_reader import scannet_scene_reader, ScanNetPPReader

processes = 2



# def get_gpu_memory_usage():
#     nvidia_smi.nvmlInit()
#     handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
#     info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
#     return (info.used/(1024 ** 3))

def save_plot(data, ylabel, title, filename):
    plt.figure()
    plt.plot(range(len(data)), data)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def reconstruct_scene(scene,experiment_name,experiment_settings,debug,oracle, n_labels = 101):
    # iteration_times = []
    # segmentation_times = []
    reconstruction_times = []
    # combined_times = []
    gpu_memory_usage = []
    # peak_memory_usage = []
    # # des = "/home/motion/semanticmapping/visuals/maskformer_default"
    arr_des = '/work/hdd/bebg/results/visuals/{}/'.format(scene)
    # # plot_dir = os.path.join(des, 'topk')
    arr_dir = os.path.join(arr_des, f'{experiment_name}')
    technique = experiment_settings['integration']
    if(technique == "topk" or technique == "Encoded Averaging"):
        print("Here")
        k = experiment_settings['k']
        arr_dir = os.path.join(arr_des, f'{experiment_name}{k}')
    print(arr_dir)
    # # if not os.path.exists(plot_dir):
    # #     os.makedirs(plot_dir)
    if not os.path.exists(arr_dir):
        print("Here2")
        os.makedirs(arr_dir)


    EG = Experiment_Generator(n_labels=n_labels)
    # EG = Experiment_Generator(n_labels=21)
    fnames = get_filenames()
    rec,model = EG.get_reconstruction_and_model(experiment = experiment_settings,process_id = multiprocessing.current_process()._identity[0])
    # print(experiment_settings)
    if(experiment_settings['segmentation'] == "CLIP"):
        get_semantics = model.get_pixel_features
    else:
        if(technique == 'Generalized'):
            get_semantics = model.get_raw_logits
        # elif(technique == 'Histogram'):
        #     get_semantics = model.classify
        else:
            get_semantics = model.get_pred_probs

    # if(not debug):
    #     root_dir = "/tmp/scannet_v2"
    # else:
    #     root_dir = "/scratch/bbuq/jcorreiamarques/3d_calibration/scannet_v2"
    root_dir = fnames['ScanNetpp_root_dir']
    # root_dir = fnames['ScanNet_root_dir']
    if experiment_settings['segmentation'] == "Maskformer":
        fnames['results_dir'] = "/work/hdd/bebg/results/scannetpp_ade20k"
    elif experiment_settings['segmentation'] == "FineMaskformer":
        fnames['results_dir'] = "/work/hdd/bebg/results/scannetpp"

    savedir = "{}/{}/".format(fnames['results_dir'], experiment_name)
    
    if(technique == "topk" or technique == "Encoded Averaging" or technique == "MisraGries" or technique == "topk KH"):
        k1 = experiment_settings['k']
        savedir = "{}/{}/".format(fnames['results_dir'], f'{experiment_name}{k}')
    
    
    # savedir = '/scratch/bbuq/jcorreiamarques/3d_calibration/Results/{}/'.format(experiment_name)
    if(not os.path.exists(savedir)):
        try:
            os.mkdir(savedir)
        except Exception as e:
            print(e)
    if debug:
        lim = -1
    else:
        lim = -1
    # pdb.set_trace()
    folder = '{}/{}'.format(savedir,scene)
    if not os.path.exists(folder):
        try:
            os.mkdir(folder)
        except Exception as e:
            print(e)
    try:
        device = o3d.core.Device('CUDA:0')

        

        # my_ds = scannet_scene_reader(root_dir, scene ,lim = lim,disable_tqdm = True)
        my_ds = ScanNetPPReader(root_dir, scene)
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
            
            # segmentation_start_time = time.time()
            semantic_label = get_semantics(data_dict['color'],depth = data_dict['depth'],x = depth.rows,y = depth.columns)
            # segmentation_end_time = time.time()
            # print(segmentation_end_time - segmentation_start_time)
            # segmentation_times.append(segmentation_end_time - segmentation_start_time)
            
            reconstruction_start_time = time.time()

            if(oracle):
                semantic_label_gt = cv2.resize(data_dict['semantic_label'],(depth.columns,depth.rows),interpolation= cv2.INTER_NEAREST)
                rec.update_vbg(data_dict['depth'],data_dict['intrinsics_depth'][:3,:3].astype(np.float64),
                            data_dict['pose'],semantic_label = semantic_label,semantic_label_gt = semantic_label_gt)
            else:
                rec.update_vbg(data_dict['depth'],data_dict['intrinsics_depth'][:3,:3].astype(np.float64),
                            data_dict['pose'],semantic_label = semantic_label, scene = scene)
            
            reconstruction_end_time = time.time()
            reconstruction_times.append(reconstruction_end_time - reconstruction_start_time)
            # combined_times.append(reconstruction_end_time - segmentation_start_time)
            # end_time = time.time()
            # iteration_times.append(end_time - start_time)
            # gpu_memory_usage.append(get_gpu_memory_usage())
            # gpu_memory_usage_np = np.array(gpu_memory_usage)
            # np.save(os.path.join(arr_dir, "gpu_memory_usage.npy"), gpu_memory_usage_np)
            reconstruction_times_np = np.array(reconstruction_times)
            # np.save(os.path.join(arr_dir, "reconstruction_times.npy"), reconstruction_times_np)
            del intrinsic
            del depth
        # save_plot(gpu_memory_usage, 'Memory Usage (GB)', 'GPU Memory Usage Over Iterations', os.path.join(plot_dir, 'gpu_memory_usage.png'))
        # gpu_memory_usage_np = np.array(gpu_memory_usage)
        # np.save(os.path.join(arr_dir, "gpu_memory_usage.npy"), gpu_memory_usage_np)

        # # Save time taken per iteration plot
        # save_plot(iteration_times, 'Time Taken (s)', 'Time Taken Per Iteration', os.path.join(plot_dir, 'iteration_times.png'))
        # iteration_times_np = np.array(iteration_times)
        # np.save(os.path.join(arr_dir, "iteration_times.npy"), iteration_times_np)

        # # Save segmentation times plot
        
        # save_plot(segmentation_times, 'Time (s)', 'Segmentation Time Per Iteration', os.path.join(plot_dir, 'segmentation_times.png'))
        # segmentation_times_np = np.array(segmentation_times)
        # np.save(os.path.join(arr_dir, "segmentation_times.npy"), segmentation_times_np)

        # # Save reconstruction times plot
        # save_plot(reconstruction_times, 'Time (s)', 'Reconstruction Time Per Iteration', os.path.join(plot_dir, 'reconstruction_times.png'))
        


        # # Save combined times plot
        # save_plot(combined_times, 'Time (s)', 'Combined Time Per Iteration', os.path.join(plot_dir, 'combined_times.png'))
        # combined_times_np = np.array(combined_times)
        # np.save(os.path.join(arr_dir, "combined_times.npy"), combined_times_np)


        # Save peak memory usage plot
        
        pcd,labels = rec.extract_point_cloud(return_raw_logits = False)
        o3d.io.write_point_cloud(folder+'/pcd_{:05d}.pcd'.format(idx), pcd, write_ascii=False, compressed=True, print_progress=False)
        pickle.dump(labels,open(folder+'/labels_{:05d}.p'.format(idx),'wb'))
        # peak_memory_usage.append(get_gpu_memory_usage())
        # peak_memory_usage_np = np.array(peak_memory_usage)
        # np.save(os.path.join(arr_dir, "peak_memory_usage.npy"), peak_memory_usage_np)



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
    parser.add_argument("--integration", type=str, required=True, help="Integration method")
    parser.add_argument("--segmentation", type=str, required=True, help="Segmentation method")
    parser.add_argument("--k", type=int, required=True, help="Top-k value")
    parser.add_argument("--num_labels", type=int, default=101, help="Number of labels")
    

    parser.add_argument('--debug', action='store_true')
    parser.set_defaults(debug = False)
    parser.add_argument('--start', type=int, default=0,
                        help="""starting Reconstruction""")
    parser.add_argument('--end', type=int, default=-1,
                        help="""starting Reconstruction""")
    args = parser.parse_args()
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

        
    # experiments = get_experiments()

    # if(args.end == -1):
    #     experiments_to_do = experiments[args.start:]
    # else:
    #     experiments_to_do = experiments[args.start:args.end]
    experiment_name = f"{args.segmentation} {args.integration}"
    print(experiment_name)
    experiments_to_do = [experiment_name]
    print('\n\n reconstructing {}\n\n'.format(experiments_to_do))
    for experiment in experiments_to_do:
        print(experiment)
        experiment_name = experiment
        # experiment_settings = json.load(open('../settings/reconstruction_experiment_settings/{}.json'.format(experiment),'rb'))
        experiment_settings.update({'experiment_name':experiment_name})
        import multiprocessing
        debug = args.debug
        oracle = experiment_settings['oracle']
        val_scenes,test_scenes = get_larger_test_and_validation_scenes()
        selected_scenes = sorted(test_scenes)
        test_scenes1 = get_small_test_scenes2()
        # dump, test_scenes1 = get_fixed_train_and_val_splits()
        selected_scenes1 = sorted(test_scenes1)
        test_scenes_pp = get_scannetpp_test_scenes()
        selected_scenes_pp = sorted(test_scenes_pp)
        p = multiprocessing.get_context('forkserver').Pool(processes = processes,maxtasksperchild = 1)

        res = []
        for a in tqdm(p.imap_unordered(partial(reconstruct_scene,experiment_name = experiment_name,experiment_settings=experiment_settings,debug = debug,oracle = oracle, n_labels=num_labels),selected_scenes_pp,chunksize = 1), total= len(selected_scenes_pp),position = 0,desc = 'tot_scenes'):
                res.append(a)

        
        
            # Save GPU memory usage plot
        

        torch.cuda.empty_cache()
        o3d.core.cuda.release_cache()
    


if __name__=='__main__':
    main()