import xmlrpc.server
import threading
import time
import queue
import os
import numpy as np
import sys
import open3d as o3d
import open3d.core as o3c
import yaml


parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from reconstruction import Reconstruction
from utils.segmentation_model_loader import TSegmenter, FineTunedTSegmenter, MaskformerSegmenter
from utils.sens_reader import scannet_scene_reader, ScanNetPPReader


class MyServer:
    def __init__(self):
        # Set up the XML-RPC server
        self.server = xmlrpc.server.SimpleXMLRPCServer(('localhost', 5001))
        self.server.register_introspection_functions()
        self.server.register_function(self.start_task)
        self.server.register_function(self.stop_task)
        self.server.register_function(self.get_map)
        self.server.register_function(self.pause_task)
        self.server.register_function(self.resume_task)
        self.server.register_function(self.get_map_stop_task)

        self.task_thread = None
        self.queue_thread = None
        self.task_running = False
        self.pause_mapping = True
        self.pause_integration = True
        self.vbg_access_lock = threading.Lock()  
        self.queue_empty = threading.Condition()

        # Dataset parameters
        root_dir = "/home/motion/Data/scannet_v2"
        scene = "scene0343_00"  
        lim = -1
        self.my_ds = scannet_scene_reader(root_dir, scene, lim=lim, disable_tqdm=True)
        self.total_len = len(self.my_ds)

        # Initialize queue and indices
        self.index_queue = 0
        self.index_reconstruction = 0
        self.queue = queue.Queue(maxsize=1000)  # Thread-safe queue
        self.load_config()


        
        


    def load_config(self):
        # Read configuration values from the YAML file
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        # Store values from the config into instance variables
        self.voxel_size = config.get('voxel_size', 0.025)
        self.trunc = self.voxel_size * config.get('truncation_vsize_multiple', 8)
        self.res = config.get('res', 8)
        self.n_labels = config.get('n_labels', 150)
        self.depth_scale = config.get('depth_scale', 1000.0)
        self.depth_max = config.get('depth_max', 5.0)
        self.miu = config.get('miu', 0.001)

        print(f"Configuration Loaded: {config}")


    def start_task(self):
        if not self.task_running:
            self.task_running = True
            self.pause_mapping = False
            self.pause_integration = False
            # Start mapping and queue threads
            self.segmenter = MaskformerSegmenter()
            self.rec = Reconstruction(
                depth_scale=self.depth_scale,
                depth_max=self.depth_max,
                res=self.res,
                voxel_size=self.voxel_size,
                n_labels=self.n_labels,
                integrate_color=False,
            )
            self.task_thread = threading.Thread(target=self.mapping, daemon=True)
            self.queue_thread = threading.Thread(target=self.fill_queue, daemon=True)
            self.queue_thread.start()
            self.task_thread.start()
            return "Task started"
        else:
            return "Task is already running"

    def stop_task(self):
        if self.task_running:
            self.task_running = False
            self.pause_mapping = True
            if self.task_thread.is_alive():
                with self.queue_empty:
                    self.queue_empty.notify()
                self.task_thread.join()  # Wait for mapping thread to finish
                


            # Safely access and delete `rec`
            with self.vbg_access_lock:
                del self.rec
                del self.segmenter
                self.segmenter = None
                self.rec = None
                self.index_queue = 0
                self.index_reconstruction = 0
                
            o3d.core.cuda.release_cache()
            print("stopping mapping!")
            return 1
        else:
            print("No mapping task was running!")
            return 0

    def get_map_stop_task(self):
        # self.pause_integration = True
        ret = self.get_map()
        self.stop_task()
        return ret


    def pause_task(self):
        self.pause_mapping = True
        return "Mapping paused"

    
    def resume_task(self):
        self.pause_mapping = False
        return "Mapping resumed"


    def mapping(self):
        while self.task_running and not self.pause_integration and (self.index_reconstruction < self.total_len):
            # print("Checking queue...")  # Added for debugging
            with self.queue_empty:
                while self.queue.empty():
                    print("waiting for frames")
                    self.queue_empty.wait()
                    if self.queue.empty(): # if queue is still empty, stop+task has notified the condition so break
                        break

                if self.queue.empty():
                    continue
                    
            data_dict = self.queue.get()
            print(f"Processing frame {self.index_reconstruction}...")

            # Ensure safe access to `rec`
            with self.vbg_access_lock:
                self.update_rec(
                    data_dict['color'],
                    data_dict['depth'],
                    data_dict['pose'],
                    data_dict['intrinsics_depth'][:3, :3].astype(np.float64)
                )
            self.index_reconstruction += 1
            self.queue.task_done()


    def fill_queue(self):
        while self.task_running and (self.index_queue < self.total_len):
            # Get a data packet from the dataset
            data_dict = self.my_ds[self.index_queue]  # Access dataset by index
            with self.queue_empty:
                if not self.pause_mapping and not self.queue.full():
                    self.queue.put(data_dict)
                    self.queue_empty.notify()
                
            self.index_queue += 1
            time.sleep(0.5)
            # waiting for mapping to catchup
            if self.index_queue % 100 == 50:
                print(f"Added frame {self.index_queue} to the queue.")
                

    def get_map(self):
        # self.pause_task()  # Pause mapping to avoid concurrent writes
        # print("here")
        with self.vbg_access_lock:
            # print("here2")
            if self.rec is not None:
                pcd, labels = self.rec.extract_point_cloud(return_raw_logits=False)
                points = np.asarray(pcd.points).tolist()
                labels = labels.tolist()
                result = {'points': points, 'labels': labels}
            else:
                result = "Reconstruction object is not initialized"
        # self.resume_task()  # Resume mapping after safe access
        return result


    def update_rec(self, rgb, depth, pose, intrinsics):
        # Perform segmentation and update reconstruction
        semantic_label = self.segmenter.get_pred_probs(
            rgb, depth, x=depth.shape[0], y=depth.shape[1]
        )
        self.rec.update_vbg(
            depth,
            intrinsics,
            pose,
            semantic_label=semantic_label
        )

    def serve_forever(self):
        # Run the server to handle incoming requests
        self.server.serve_forever()

if __name__ == '__main__':
    server = MyServer()
    server.serve_forever()
