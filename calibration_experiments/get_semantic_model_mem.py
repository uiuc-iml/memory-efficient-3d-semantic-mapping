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


# from experiment_setup import Experiment_Generator
# from utils.ScanNet_scene_definitions import get_filenames, get_larger_test_and_validation_scenes, get_smaller_test_scenes, get_small_test_scenes2, get_fixed_train_and_val_splits, get_scannetpp_test_scenes
# from utils.sens_reader import scannet_scene_reader, ScanNetPPReader
from utils.segmentation_model_loader import TSegmenter,FineTunedTSegmenter, MaskformerSegmenter


processes = 1



def get_gpu_memory_usage():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return (info.used/(1024 ** 3))

seg = MaskformerSegmenter()

# Load your image (replace with your image loading logic)
image = cv2.imread("im.png")

# List to store memory usage for each iteration
memory_usage = []

# Number of iterations for averaging
num_iterations = 100

for _ in range(num_iterations):
    # Get GPU memory usage before segmentation
    start_memory = get_gpu_memory_usage()

    # Perform segmentation
    _ = seg.get_raw_logits(image)  # Replace with the appropriate segmentation function

    # Get GPU memory usage after segmentation
    end_memory = get_gpu_memory_usage()

    # Calculate memory usage for this iteration
    memory_usage.append(end_memory - start_memory)

# Calculate and display mean memory usage
mean_memory = np.mean(memory_usage)
print(f"Mean GPU memory usage during segmentation: {mean_memory:.2f} GB")
