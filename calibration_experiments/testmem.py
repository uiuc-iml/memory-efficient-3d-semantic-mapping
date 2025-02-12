import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.dlpack
import gc
import sys

import lightning as L
from litautoencoder import LitAutoEncoder
import numpy as np
import open3d as o3d
import open3d.core as o3c
from tqdm import tqdm
import cv2
from klampt.math import se3
import pickle
import torch
import pdb
import torch.utils.dlpack
from torch import linalg as LA
import torch.nn as nn
import os
import nvidia_smi
import matplotlib.pyplot as plt
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from utils.segmentation_model_loader import TSegmenter,FineTunedTSegmenter, MaskformerSegmenter


def get_gpu_memory_usage():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return (info.used/(1024 ** 3))



class Encoder(nn.Module):
    def __init__(self, input_dim, encoded_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, encoded_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Define the decoder MLP with four layers
class Decoder(nn.Module):
    def __init__(self, encoded_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(encoded_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.softmax(x, dim=-1)
        return x


torch.cuda.memory._record_memory_history(max_entries=100000)


print(f' total memory allocated before initializing cuda: {torch.cuda.memory_allocated()}')
print(f' total memory reserved before initializing cuda: {torch.cuda.memory_reserved()}')
print(f'gpu memory: {get_gpu_memory_usage()} GB')
torch.empty((1,1), device='cuda')
print(f' total memory allocated after initializing cuda: {torch.cuda.memory_allocated()}')
print(f' total memory reserved after initializing cuda: {torch.cuda.memory_reserved()}')
initial_memory0 = get_gpu_memory_usage()
print(f"Initial memory usage before loading model: {initial_memory0:.4f} GB")

torch.cuda.empty_cache()


# # Create an instance of the Encoder model
input_dim = 150
encoded_dim = 4
encoder_model = Encoder(input_dim, encoded_dim)

# # for name, param in encoder_weights.items():
# #     print(f"{name}: {param.shape}")

# # Load the extracted weights into the bare Encoder model
encoder_weights_loaded = torch.load('encoder_weights.pt', weights_only=True)
# # print((encoder_weights_loaded))
# # print((encoder_weights_loaded['fc1.weight']).dtype)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # t = torch.Tensor([1]).to(device)
initial_memory2 = get_gpu_memory_usage()
print(f"Initial memory usage after loading weights: {initial_memory2:.4f} GB")
print(torch.cuda.memory_allocated())
gc.collect()

torch.cuda.empty_cache()

print(torch.cuda.memory_reserved())
# # print(f"Initial memory usage after loading weights: {initial_memory2:.4f} GB")
# # print("Current Memory cached occupied by tensors: ", torch.cuda.memory_cached())


# # torch.cuda.set_per_process_memory_fraction(0.01)  # Sets the per-process memory fraction to 100% (default behavior)


encoder_model.load_state_dict(encoder_weights_loaded)
del encoder_weights_loaded
torch.cuda.empty_cache()




# # Move the model to the GPU
encoder_model.to(device)
encoder_model = torch.compile(encoder_model, "reduce-overhead")
# # Set the model to evaluation mode
# # encoder_model.eval()

# for param_tensor in encoder_model.state_dict():
#     print(param_tensor, "\t", encoder_model.state_dict()[param_tensor].size())


# # Print the number of trainable parameters (weights) in the encoder model


# # Initialize memory tracking list
memory_usage = []

# # Batch size
batch_size = 2000
num_batches = 50  # Adjust as necessary for your testing

# # Get initial memory usage
initial_memory = get_gpu_memory_usage()
print(f"Initial memory usage: {initial_memory:.4f} GB")
# # print(torch.cuda.memory_summary())

# segm = MaskformerSegmenter()
# # Tracking memory usage for each iteration
for i in range(num_batches):
    # Generate Open3D tensor on CUDA device
    encodeinput = o3c.Tensor(np.random.rand(batch_size, input_dim).astype(np.float32), device=o3c.Device('CUDA:0'))
    

    # Convert Open3D tensor to PyTorch tensor
    encodeinput_t = torch.utils.dlpack.from_dlpack(encodeinput.to_dlpack()).cuda()

    # Perform encoding without gradient tracking
    with torch.no_grad(), torch.cuda.amp.autocast():
        # Perform the encoding operation
        encoded_obs_t = encoder_model(encodeinput_t)

        # Measure GPU memory usage after the operation
        memory_used = get_gpu_memory_usage()
        memory_usage.append(memory_used)

    # Clear unnecessary cached memory
    del encodeinput, encodeinput_t, encoded_obs_t  
    torch.cuda.empty_cache()
    o3d.core.cuda.release_cache()

torch.cuda.memory._dump_snapshot("profile.pkl")
torch.cuda.memory._record_memory_history(enabled=None)

# Plot memory consumption over iterations
memory_usage_np = np.array(memory_usage)

# Save memory usage as a NumPy file
np.save('memory_usage.npy', memory_usage_np)

# Plot memory consumption over iterations
plt.figure(figsize=(10, 6))
plt.plot(range(num_batches), memory_usage_np, label="GPU Memory Usage (GB)", color='blue')
plt.xlabel("Batch")
plt.ylabel("Memory Usage (GB)")
plt.title("Memory Consumption Over Batches")
plt.legend()
plt.grid(True)
plt.show()

# # Calculate mean memory usage during the iterations
mean_memory_usage = np.mean(memory_usage_np)
print(f"Mean memory usage during batches: {mean_memory_usage:.4f} GB")

# # Calculate and print the difference between mean memory usage and initial memory
memory_usage_difference = mean_memory_usage - initial_memory
print(f"Difference in memory usage (mean - initial): {memory_usage_difference:.4f} GB")




