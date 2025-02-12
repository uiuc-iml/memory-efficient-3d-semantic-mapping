import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

import torch
from copy import deepcopy
from PIL import Image
import cv2
import h5py
import numpy as np
from tqdm import tqdm
import open_clip
from utils.ScanNet_scene_definitions import get_filenames, get_fixed_train_and_val_splits, h5pyscenes, get_small_test_scenes2, get_larger_test_and_validation_scenes
from utils.sens_reader import scannet_scene_reader, ScanNetPPReader
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize the SAM model and mask generator
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(device)
sam.eval()
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=8,
    pred_iou_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
)
def check_overlap(mask1, mask2):

    # print(mask1)
    overlap = np.logical_and(mask1['segmentation'], mask2['segmentation'])
    overlap = overlap.reshape(-1)
    return np.sum(overlap)

def get_total_frames(scene_names, root_dir, sample_interval=25):
    total_frames = 0
    for scene_name in scene_names:
        scene = scannet_scene_reader(root_dir, scene_name)
        total_frames += len(scene) // sample_interval
    return total_frames

def create_dataset(scene_names, root_dir, output_dir, sample_interval=25, chunk_size=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # total_frames = get_total_frames(scene_names, root_dir, sample_interval)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k", device=device)
    model.eval()

    hdf5_path = 'embeddingst_dataset.h5'
    with h5py.File(hdf5_path, 'w') as f:
        dataset = f.create_dataset(
            'embeddings', shape=(2604300, 1024), # shape=((total_frames+1)*150, 1024)
            maxshape=(None, 1024), dtype=np.float32,
            chunks=(chunk_size, 1024), compression='gzip'
        )
        current_frame = 0

        for scene_name in tqdm(scene_names, desc="Processing scenes"):
            scene = scannet_scene_reader(root_dir, scene_name)
            for i in tqdm(range(0, len(scene), sample_interval), desc=f"Processing frames in {scene_name}"):
                frame = scene[i]
                color_image = frame['color']
                image_np = deepcopy(color_image)
                rgb = Image.fromarray(color_image.astype('uint8'), 'RGB')

                with torch.no_grad(), torch.cuda.amp.autocast():
                    masks = mask_generator.generate(np.array(rgb))
                    # for i in range(len(masks)):
                    #     for j in range(len(masks)):
                    #         if i != j:
                    #             print(check_overlap(masks[i], masks[j]))
                
                    # print("we're so back")
                    _img = preprocess(rgb).unsqueeze(0).to(device)
                    global_feat = model.encode_image(_img)
                    global_feat /= global_feat.norm(dim=-1, keepdim=True)
                    torch.cuda.empty_cache()

                similarity_scores, feat_per_roi, roi_nonzero_inds = [], [], []
                with torch.no_grad(), torch.cuda.amp.autocast():
                    for maskidx in range(len(masks)):
                        _x, _y, _w, _h = tuple(masks[maskidx]["bbox"])
                        _y = (int)(_y)
                        _x = (int)(_x)
                        _w = (int)(_w)
                        _h = (int)(_h)
                        seg = masks[maskidx]["segmentation"]
                        nonzero_inds = torch.argwhere(torch.from_numpy(seg))
                        img_roi = np.array(rgb)[_y:_y + _h + 1, _x:_x + _w + 1, :]
                        img_roi = Image.fromarray(img_roi)
                        img_roi = preprocess(img_roi).unsqueeze(0).to(device)
                        
                        roifeat = model.encode_image(img_roi)
                        roifeat /= roifeat.norm(dim=-1)

                        feat_per_roi.append(roifeat)
                        roi_nonzero_inds.append(nonzero_inds)
                        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
                        _sim = cosine_similarity(global_feat, roifeat)
                        similarity_scores.append(_sim)
                        torch.cuda.empty_cache()

                    similarity_scores = torch.cat(similarity_scores)
                    softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)

                    outfeat = torch.zeros(image_np.shape[0], image_np.shape[1], 1024, dtype=torch.float, device=device)
                    for maskidx in range(len(masks)):
                        _weighted_feat = (
                            softmax_scores[maskidx] * global_feat
                            + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
                        )
                        _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
                        outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] += _weighted_feat[0].detach()
                        outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] = torch.nn.functional.normalize(
                            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]].float(), dim=-1)
                    
                    height, width = outfeat.shape[0], outfeat.shape[1]
                    grid_x, grid_y = torch.meshgrid(
                        torch.linspace(0, width - 1, steps=10),
                        torch.linspace(0, height - 1, steps=10))

                    height, width = outfeat.shape[0], outfeat.shape[1]
                    grid_x, grid_y = torch.meshgrid(
                        torch.linspace(0, width - 1, steps=10),
                        torch.linspace(0, height - 1, steps=10))

                    # Convert to integers (to ensure valid indexing)
                    grid_points = torch.stack([grid_y.flatten().long(), grid_x.flatten().long()], dim=-1).to(device)

                    # Filter out grid points with zero embeddings
                    valid_grid_points = []
                    for pt in grid_points:
                        if torch.norm(outfeat[pt[0], pt[1]]) != 0:
                            valid_grid_points.append(pt)
                    valid_grid_points = torch.stack(valid_grid_points)

                    # Now, sample non-zero embeddings from the grid points
                    non_zero_inds = torch.argwhere(outfeat.sum(dim=-1) != 0)
                    sampled_inds = non_zero_inds[torch.randperm(len(non_zero_inds))[:50]]

                    # Combine grid points and random samples
                    final_sampled_inds = torch.cat([valid_grid_points, sampled_inds], dim=0)

                    # Ensure we have exactly 150 samples
                    if final_sampled_inds.shape[0] < 150:
                        extra_samples_needed = 150 - final_sampled_inds.shape[0]
                        extra_samples = non_zero_inds[torch.randperm(len(non_zero_inds))[:extra_samples_needed]]
                        final_sampled_inds = torch.cat([final_sampled_inds, extra_samples], dim=0)

                    # Get the corresponding embeddings from outfeat
                    sampled_embeddings = outfeat[final_sampled_inds[:, 0], final_sampled_inds[:, 1]].cpu().numpy()

                    # sampled_embeddings = outfeat[sampled_inds[:, 0], sampled_inds[:, 1]].cpu().numpy()

                    # outfeat = outfeat.reshape(50, 1024)
                    dataset[current_frame:current_frame+150] = sampled_embeddings
                    current_frame += 150

    print(f"Dataset saved to {hdf5_path}")

def view_h5_file(h5_file):
    with h5py.File(h5_file, 'r') as f:
        print("Keys in the HDF5 file:", list(f.keys()))
        images = f['embeddings']
        print("Shape of images dataset:", images.shape)

        device = 'cuda:0'
        sample_image = images[2604299]
        print(np.sum(np.any(images, axis=1)))
        np.set_printoptions(threshold=np.inf)
        print(sample_image[np.argmax(sample_image)])

        print(sample_image)
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k", device=device)
        model.eval()
        with torch.no_grad():
            tokenizer = open_clip.get_tokenizer('ViT-H-14')
            text2 = tokenizer(["a picture of literally anything"])
            text_features2 = model.encode_text(text2.to(device))
            text_features2 /= text_features2.norm(dim=-1, keepdim=True)
            sim2 = sample_image @ text_features2.cpu().numpy().T
            print(sim2)

if __name__ == '__main__':
    scene_names, _ = get_larger_test_and_validation_scenes()
    # scene_names = [get_small_test_scenes2()[0]]
    print(scene_names)
    # scene_names = ["scene0077_00"]
    fnames = get_filenames()
    root_dir = fnames['ScanNet_root_dir']
    output_dir = "output_dataset"
    # create_dataset(scene_names, root_dir, output_dir, sample_interval=10, chunk_size=100)
    
    h5_file = 'embeddingst_dataset.h5'
    view_h5_file(h5_file)
