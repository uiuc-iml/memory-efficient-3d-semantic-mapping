from PIL import Image
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator  # Import the generator
import cv2  # For color manipulation
import open_clip
from copy import deepcopy

# class ClipFeatureExtractor():
#     def __init__(self):
#         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         self.sam = (sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")).to(self.device) # Ensure you have the correct checkpoint
#         # sam.to("cuda:0")
#         self.sam.eval()  # Set the model to evaluation mode
#         self.mask_generator = SamAutomaticMaskGenerator(
#                                 model=sam,
#                                 points_per_side=8,
#                                 pred_iou_thresh=0.92,
#                                 crop_n_layers=1,
#                                 crop_n_points_downscale_factor=2,)
#         self.model, _, self.preprocess = open_clip.create_model_and_transforms(
#         "ViT-H-14", "laion2b_s32b_b79k")
#         self.model.cuda()
#         self.model.eval()

#     def get_pixel_features(self, rgb,depth = None,x=None,y = None):

class ClipFeatureExtractor:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.sam = (sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")).to(self.device)
        self.sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=8,
            pred_iou_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k", device=self.device
        )
        self.model.eval()

    def get_pixel_features(self, rgb, depth=None, x=None, y=None):
        # Ensure the RGB image is in the right format
        # if isinstance(rgb, str):
        #     rgb = Image.open(rgb).convert("RGB")
        image_np = deepcopy(rgb)
        if isinstance(rgb, np.ndarray):
            # Convert NumPy array to PIL Image
            rgb = Image.fromarray(rgb.astype('uint8'), 'RGB')


        # Generate masks
        

        # Extract global features
        with torch.no_grad(), torch.cuda.amp.autocast():
            masks = self.mask_generator.generate(np.array(rgb))
            _img = self.preprocess(rgb).unsqueeze(0).to(self.device)
            global_feat = self.model.encode_image(_img)
            global_feat /= global_feat.norm(dim=-1, keepdim=True)
            torch.cuda.empty_cache()

        similarity_scores = []
        feat_per_roi = []
        roi_nonzero_inds = []

        # Process each mask
        with torch.no_grad(), torch.cuda.amp.autocast():
            for maskidx in range(len(masks)):
                _x, _y, _w, _h = tuple(masks[maskidx]["bbox"])
                # print(masks[maskidx]["bbox"])
                seg = masks[maskidx]["segmentation"]
                nonzero_inds = torch.argwhere(torch.from_numpy(seg))

                # Extract the ROI
                img_roi = image_np[_y:_y + _h + 1, _x:_x + _w + 1, :]
                img_roi = Image.fromarray(img_roi)
                img_roi = self.preprocess(img_roi).unsqueeze(0).to(self.device)

                
                roifeat = self.model.encode_image(img_roi)
                roifeat /= roifeat.norm(dim=-1)

                feat_per_roi.append(roifeat)
                roi_nonzero_inds.append(nonzero_inds)

                # Calculate cosine similarity
                cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
                _sim = cosine_similarity(global_feat, roifeat)
                similarity_scores.append(_sim)
                # print(maskidx)
                # print(torch.cuda.memory_allocated())
                torch.cuda.empty_cache()

        # Convert similarity scores to tensor
        similarity_scores = torch.cat(similarity_scores)
        softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)

        # Prepare the output features
        feat_dim = global_feat.shape[-1]
        outfeat = torch.zeros(image_np.shape[0], image_np.shape[1], feat_dim, dtype=torch.half, device=self.device)

        for maskidx in range(len(masks)):
            _weighted_feat = softmax_scores[maskidx] * global_feat + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
            _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)

            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] += _weighted_feat[0].detach()
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] = torch.nn.functional.normalize(
                outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]].float(), dim=-1)

        return outfeat.float().detach().contiguous().cpu().numpy()








































# from PIL import Image
# import torch
# import numpy as np
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator  # Import the generator
# import cv2  # For color manipulation
# import open_clip

# # class ClipFeatureExtractor():
# #     def __init__(self):
# #         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
# #         self.sam = (sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")).to(self.device) # Ensure you have the correct checkpoint
# #         # sam.to("cuda:0")
# #         self.sam.eval()  # Set the model to evaluation mode
# #         self.mask_generator = SamAutomaticMaskGenerator(
# #                                 model=sam,
# #                                 points_per_side=8,
# #                                 pred_iou_thresh=0.92,
# #                                 crop_n_layers=1,
# #                                 crop_n_points_downscale_factor=2,)
# #         self.model, _, self.preprocess = open_clip.create_model_and_transforms(
# #         "ViT-H-14", "laion2b_s32b_b79k")
# #         self.model.cuda()
# #         self.model.eval()

# #     def get_pixel_features(self, rgb,depth = None,x=None,y = None):

# class ClipFeatureExtractor:
#     def __init__(self):
#         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         self.sam = (sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")).to(self.device)
#         self.sam.eval()
#         self.mask_generator = SamAutomaticMaskGenerator(
#             model=self.sam,
#             points_per_side=8,
#             pred_iou_thresh=0.92,
#             crop_n_layers=1,
#             crop_n_points_downscale_factor=2,
#         )
#         self.model, _, self.preprocess = open_clip.create_model_and_transforms(
#             "ViT-H-14", "laion2b_s32b_b79k", device=self.device
#         )
#         self.model.eval()

#     def get_pixel_features(self, rgb, depth=None, x=None, y=None):
#         # Ensure the RGB image is in the right format
#         if isinstance(rgb, str):
#             rgb = Image.open(rgb).convert("RGB")
#         image_np = np.array(rgb)

#         # Generate masks
#         masks = self.mask_generator.generate(image_np)

#         # Extract global features
#         with torch.no_grad(), torch.cuda.amp.autocast():
#             _img = self.preprocess(rgb).unsqueeze(0).to(self.device)
#             global_feat = self.model.encode_image(_img)
#             global_feat /= global_feat.norm(dim=-1, keepdim=True)

#         similarity_scores = []
#         feat_per_roi = []
#         roi_nonzero_inds = []

#         # Process each mask
#         for maskidx in range(len(masks)):
#             _x, _y, _w, _h = tuple(masks[maskidx]["bbox"])
#             seg = masks[maskidx]["segmentation"]
#             nonzero_inds = torch.argwhere(torch.from_numpy(seg))

#             # Extract the ROI
#             img_roi = image_np[_y:_y + _h, _x:_x + _w, :]
#             img_roi = Image.fromarray(img_roi)
#             img_roi = self.preprocess(img_roi).unsqueeze(0).to(self.device)

#             with torch.no_grad(), torch.cuda.amp.autocast():
#                 roifeat = self.model.encode_image(img_roi)
#                 roifeat /= roifeat.norm(dim=-1)

#             feat_per_roi.append(roifeat)
#             roi_nonzero_inds.append(nonzero_inds)

#             # Calculate cosine similarity
#             cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
#             _sim = cosine_similarity(global_feat, roifeat)
#             similarity_scores.append(_sim)

#         # Convert similarity scores to tensor
#         similarity_scores = torch.cat(similarity_scores)
#         softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)

#         # Prepare the output features
#         feat_dim = global_feat.shape[-1]
#         outfeat = torch.zeros(image_np.shape[0], image_np.shape[1], feat_dim, dtype=torch.half, device=self.device)

#         for maskidx in range(len(masks)):
#             _weighted_feat = softmax_scores[maskidx] * global_feat + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
#             _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)

#             outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] += _weighted_feat[0].detach()
#             outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] = torch.nn.functional.normalize(
#                 outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]].float(), dim=-1
#             ).half()

#         return outfeat.float()
