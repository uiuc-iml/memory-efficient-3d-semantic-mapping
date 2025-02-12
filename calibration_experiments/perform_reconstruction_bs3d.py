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
from tkinter import E
import psutil
import time
import torch
import nvidia_smi

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
from utils.sens_reader import scannet_scene_reader, ScanNetPPReader, BS3D_reader

processes = 1



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


def reconstruct_scene(scene,experiment_name,experiment_settings,debug,oracle):
    # iteration_times = []
    # segmentation_times = []
    reconstruction_times = []
    # combined_times = []
    gpu_memory_usage = []
    # peak_memory_usage = []
    # # des = "/home/motion/semanticmapping/visuals/maskformer_default"
    arr_des = '/home/motion/semanticmapping/visuals/arrays/{}/cacherelease'.format(scene)
    # # plot_dir = os.path.join(des, 'topk')
    arr_dir = os.path.join(arr_des, f'{experiment_name}')
    # arr_dir = os.path.join(arr_des, f'scannetpp_Segformer_150_topk1')
    # # if not os.path.exists(plot_dir):
    # #     os.makedirs(plot_dir)
    if not os.path.exists(arr_dir):
        os.makedirs(arr_dir)


    EG = Experiment_Generator(n_labels=150)
    # EG = Experiment_Generator(n_labels=21)
    fnames = get_filenames()
    rec,model = EG.get_reconstruction_and_model(experiment = experiment_settings,process_id = multiprocessing.current_process()._identity[0])
    if(experiment_settings['integration'] == 'Generalized'):
        get_semantics = model.get_raw_logits
    # elif(experiment_settings['integration'] == 'Histogram'):
    #     get_semantics = model.classify
    else:
        get_semantics = model.get_pred_probs

    # if(not debug):
    #     root_dir = "/tmp/scannet_v2"
    # else:
    #     root_dir = "/scratch/bbuq/jcorreiamarques/3d_calibration/scannet_v2"
    root_dir = fnames['bs3d_root_dir']
    # root_dir = fnames['ScanNet_root_dir']
    savedir = "{}/{}/".format(fnames['results_bs3d_dir'], experiment_name)
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
        my_ds = BS3D_reader(root_dir)
        total_len = len(my_ds)

        if(lim == -1):
            lim = total_len
        randomized_indices = np.array(list(range(lim)))
        np.random.seed(0)
        proc_num = multiprocessing.current_process()._identity[0]%(processes+1) + 1
        view_data = {
       "boundingbox_max" : [ 77.142349243164062, 24.75, 2.5 ],
"boundingbox_min" : [ -11.343977928161621, -216.6998291015625, -10.474475860595703 ],
"field_of_view" : 60.0,
"front" : [ -0.16450450961717239, 0.76824444671868797, 0.61865882067704392 ],
"lookat" : [ 32.899185657501221, -95.97491455078125, -3.9872379302978516 ],
"up" : [ 0.12403725235918707, -0.60611303386894388, 0.78564734467913011 ],
"zoom" : 0.19999999999999983
}



        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Set the view control once
        ctr = vis.get_view_control()
        ctr.set_front(view_data["front"])
        ctr.set_lookat(view_data["lookat"])
        ctr.set_up(view_data["up"])
        ctr.set_zoom(view_data["zoom"])
        point_size = 2.0  # Start with a default point size

        # Get render options to modify point size
        render_option = vis.get_render_option()
        render_option.point_size = point_size


        frame_width = 1850
        frame_height = 1016
        output_video = cv2.VideoWriter("output_video_ef4.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (frame_width, frame_height))
        for idx,i in tqdm(enumerate(randomized_indices),total = lim,desc = 'proc {}'.format(proc_num),position = proc_num):
            # start_time = time.time()
            
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
            del intrinsic
            del depth
            if (i < 5):
                continue
            if (i % 5 != 0):
                # break
                continue
              
            # if (i == 60):
            #     break
            pcd,toplabels = rec.extract_point_cloud(return_raw_logits = False)
            # toplabels = np.argmax(toplabels, axis=1)
            ceilingmask = np.logical_and(toplabels != 0, toplabels != 1000)
            # ceilingmask = toplabels != 0
            newtoplabels = toplabels[ceilingmask]
            points = np.asarray(pcd.points)
            filtered_points = points[ceilingmask]
            # print(filtered_points.shape)

            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            normals = np.asarray(pcd.normals)
            filtered_normals = normals[ceilingmask]
            new_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)
            bs3d2 = np.load('bs3d2.npy')
            new_colors = np.array(bs3d2)[newtoplabels.astype(int)]
            new_pcd.colors = o3d.utility.Vector3dVector(new_colors)
            vis.clear_geometries()
            

            # Apply the view parameters in each iteration
            # ctr = vis.get_view_control()
            # ctr.set_front(view_data["front"])
            # ctr.set_lookat(view_data["lookat"])
            # ctr.set_up(view_data["up"])
            # ctr.set_zoom(view_data["zoom"])

            # Render and capture screen
            vis.add_geometry(new_pcd)
            vis.poll_events()
            ctr = vis.get_view_control()
            ctr.set_front(view_data["front"])
            ctr.set_lookat(view_data["lookat"])
            ctr.set_up(view_data["up"])
            ctr.set_zoom(view_data["zoom"])
            vis.poll_events()
            vis.update_renderer()
            # print(f"Rendering frame {i+1}")
            # time.sleep(1)  # Adjust sleep time as necessary
            screenshot_path = f"screenshot_{i+1}.png"
            vis.capture_screen_image(screenshot_path)
            
            # Read the image file into OpenCV and write it to the video
            frame = cv2.imread(screenshot_path)
            if frame is not None:
                output_video.write(frame)
            else:
                print(f"Failed to read {screenshot_path}")
            os.remove(screenshot_path)


            

            # if i == 6000:
            #     break
            # try:
            #     # print(i%1000)
            #     if i%1000 == 0 and i != 0:
            #         pcd,labels = rec.extract_point_cloud_max(return_raw_logits = False)
            #         o3d.io.write_point_cloud(folder+'/pcd_{:05d}.pcd'.format(idx), pcd, write_ascii=False, compressed=True, print_progress=False)
            #         pickle.dump(labels,open(folder+'/labels_{:05d}.p'.format(idx),'wb'))
            # except Exception as e:
            #     print(e)
                
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
        reconstruction_times_np = np.array(reconstruction_times)
        np.save(os.path.join(arr_dir, "reconstruction_times.npy"), reconstruction_times_np)
        vis.destroy_window()
        output_video.release()


        # # Save combined times plot
        # save_plot(combined_times, 'Time (s)', 'Combined Time Per Iteration', os.path.join(plot_dir, 'combined_times.png'))
        # combined_times_np = np.array(combined_times)
        # np.save(os.path.join(arr_dir, "combined_times.npy"), combined_times_np)


        # Save peak memory usage plot
        
        # pcd,labels = rec.extract_point_cloud_max(return_raw_logits = False)
        # o3d.io.write_point_cloud(folder+'/pcd_{:05d}.pcd'.format(idx), pcd, write_ascii=False, compressed=True, print_progress=False)
        # pickle.dump(labels,open(folder+'/labels_{:05d}.p'.format(idx),'wb'))
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

    print('\n\n reconstructing {}\n\n'.format(experiments_to_do))
    for experiment in experiments_to_do:
        print(experiment)
        experiment_name = experiment
        experiment_settings = json.load(open('../settings/reconstruction_experiment_settings/{}.json'.format(experiment),'rb'))
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
        bs3d_scenes = ["campus"]
        p = multiprocessing.get_context('forkserver').Pool(processes = processes,maxtasksperchild = 1)

        res = []
        for a in tqdm(p.imap_unordered(partial(reconstruct_scene,experiment_name = experiment_name,experiment_settings=experiment_settings,debug = debug,oracle = oracle),bs3d_scenes,chunksize = 1), total= len(bs3d_scenes),position = 0,desc = 'tot_scenes'):
                res.append(a)

        
        
        

        torch.cuda.empty_cache()
        o3d.core.cuda.release_cache()
    


if __name__=='__main__':
    main()