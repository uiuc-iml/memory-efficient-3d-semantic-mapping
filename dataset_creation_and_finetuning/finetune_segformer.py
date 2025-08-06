import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from torch import nn
from torchvision.transforms import ColorJitter
from transformers import (
    EarlyStoppingCallback,
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation,
    Trainer,
    TrainingArguments,
)
import cv2
import pickle
import pdb
import sys,os
import time

import torch
import torch.nn.functional as F
import argparse
from multiprocessing import Lock

from datasets import IterableDataset,Dataset

parser = argparse.ArgumentParser()


parser.add_argument('--parallel',action='store_true',help ='whether this model should be trained in parallel')
parser.add_argument('--loss_weights',default = 'sqrt',help = 'how to scale the loss weights per class')
parser.add_argument('--lr',default = 0.001,type = float, help = 'the learning rate for training the model')
parser.add_argument('--gpu',default = 'A100', help = 'the GPU used for training')

args = parser.parse_args()

parallel = args.parallel
loss_weights = args.loss_weights
lr = args.lr
gpu = args.gpu
N_CLASSES = 101

# ds = IterableDataset.from_generator(data_generator)

class Cumulative_mIoU_torch:
    def __init__(self,n_classes):
        self.n_classes = n_classes
        self.reset_counts()
    
    def reset_counts(self):
        self.intersections = torch.from_numpy(np.zeros(self.n_classes)).long().to('cuda:0')
        self.unions = torch.from_numpy(np.zeros(self.n_classes)).long().to('cuda:0')
        self.all_preds =  torch.from_numpy(np.zeros(self.n_classes)).long().to('cuda:0')
        self.all_gts = torch.from_numpy(np.zeros(self.n_classes)).long().to('cuda:0')
        self.lock = Lock()
    def update_counts(self,pred,gt):
        with self.lock:
            with torch.no_grad():
                self.intersections = self.intersections.to(pred.device)
                self.unions = self.unions.to(pred.device)
                self.all_gts = self.all_gts.to(pred.device)
                self.all_preds = self.all_preds.to(pred.device)
                gt = gt.to(pred.device)
                for i in range(self.n_classes):
                    gt_mask = torch.eq(gt.long(),i)
                    pred_mask = torch.eq(pred.long(),i)
                    self.intersections[i] += torch.logical_and(gt_mask,pred_mask).sum()
                    self.unions[i] += torch.logical_or(gt_mask,pred_mask).sum()
                    self.all_preds[i] += pred_mask.sum()
                    self.all_gts[i] += gt_mask.sum()

    def get_IoUs(self):
        res = self.intersections.cpu().numpy()/self.unions.cpu().numpy()
        # res[np.logical_not(np.isfinite(res))] = 0
        return res
    def get_precision(self):
        res = self.intersections.cpu().numpy()/self.all_preds.cpu().numpy()
        # res[np.logical_not(np.isfinite(res))] = 0
        return res
    def get_recall(self):
        res = self.intersections.cpu().numpy()/self.all_gts.cpu().numpy()
        # res[np.logical_not(np.isfinite(res))] = 0
        return res



def multiclass_dice_loss(pred, tget, smooth=1):
    """
    Computes Dice Loss for multi-class segmentation. Thanks to https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
    Args:
        pred: Tensor of predictions (batch_size, C, H, W).
        target: ground truth (batch_size,H, W).
        smooth: Smoothing factor.
    Returns:
        Scalar Dice Loss.

    """
    pred = F.softmax(pred, dim=1)  # Convert logits to probabilities
    num_classes = pred.shape[1]  # Number of classes (C)
    dice = 0  # Initialize Dice loss accumulator
    if(len(tget)!=len(pred.shape)): # if the shapes don't match, the input must be one-hot-encoded
        target = nn.functional.one_hot(tget,num_classes = num_classes).permute(0,3,1,2)
    else:
        target = tget

    for c in range(num_classes):  # Loop through each class
        pred_c = pred[:, c]  # Predictions for class c

        target_c = target[:, c]  # Ground truth for class c
        
        intersection = (pred_c * target_c).sum(dim=(1, 2))  # Element-wise multiplication
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))  # Sum of all pixels
        
        dice += (2. * intersection + smooth) / (union + smooth)  # Per-class Dice score

    return (1 - dice.mean() / num_classes).contiguous()  # Average Dice Loss across classes



def load_images(rgb,semantic):
    # import pdb
    # pdb.set_trace()
    rgb_batch = []
    sem_batch = []
    for sem,color in zip(rgb,semantic):
        tmp = cv2.imread(semantic[0],cv2.IMREAD_UNCHANGED).astype(np.uint8)
        tmp2 = cv2.imread(rgb[0]).astype(np.uint8)
        tmp2 = np.ascontiguousarray(np.moveaxis(tmp2,[0,1,2],[1,2,0]))
        rgb_batch.append(tmp2)
        sem_batch.append(tmp)
    return rgb_batch,sem_batch



parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


# fnames = get_filenames()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model =  SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512").to(device)


model.train()
for param in model.parameters():
    param.requires_grad = True
decoder_head = model.decode_head
decoder_head.classifier = nn.Conv2d(768,N_CLASSES,kernel_size=(1, 1), stride=(1, 1))
for param in decoder_head.parameters():
    param.requires_grad = True

model.to(device)

# if(parallel):
#     from accelerate import Accelerator

# classes = ['irrelevant','wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture',
#            'counter','desk','curtain','refridgerator','shower curtain','toilet','sink','bathtub','otherfurniture']
# id2label = {}
# label2id = {}
# print("{")
# for i in range(len(classes)):
#     comma = ''
#     if(i!= len(classes)-1):
#         comma = ','
#     print('\"{}\":\"{}\"{}'.format(classes[i],i,comma))
#     id2label.update({i:classes[i]})
# print('}')

feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
feature_extractor.reduce_labels = False
feature_extractor.do_reduce_labels = False

train_ds_dir = './scannet_pp_finetune_train.hf'
val_ds_dir = './scannet_pp_finetune_val.hf'

train_ds = Dataset.load_from_disk(train_ds_dir)
train_ds = train_ds.select(np.arange(0,train_ds.shape[0],10))
val_ds = Dataset.load_from_disk(val_ds_dir)
val_ds = val_ds.select(np.arange(0,val_ds.shape[0],10))


# ds = Dataset.load_from_disk(huggingface_dataset_dir,keep_in_memory = True)
# ds = ds
# ds = ds.train_test_split(test_size=0.2)
# train_ds = ds["train"]


# full_test_ds = ds["test"]
# test_ds = full_test_ds.shuffle(seed = 1).train_test_split(test_size = 0.1)['test']

jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 

def train_transforms(example_batch):
#     images = [x for x in example_batch['pixel_values']]
#     labels = [x for x in example_batch['label']]
    # import pdb
    # pdb.set_trace()

    # a = np.array(example_batch['pixel_values']).astype(np.uint8)[0]
    # b = np.array(example_batch['label']).astype(np.uint8)[0]
    a,b = load_images(example_batch['pixel_values'],example_batch['label'])

    inputs = feature_extractor(a,b)
    # pdb.set_trace()
    return inputs


def val_transforms(example_batch):
#     images = [x for x in example_batch['pixel_values']]
#     labels = [x for x in example_batch['label']]
#     inputs = feature_extractor(images, labels)
    a,b = load_images(example_batch['pixel_values'],example_batch['label'])
    inputs = feature_extractor(a,b)
    
    return inputs

# Set transforms
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)

train_ds = train_ds.shuffle(seed = 32)


# set of weights with laplace smoothing

ops = {'sqrt':np.sqrt,'log':np.log,'abs':np.abs}

op = ops.get(loss_weights,np.abs)

weights = op(np.array([3.5929339e+00, 4.3877968e+01, 6.9125867e+00, 1.0403013e+01,
       4.3412052e+01, 3.5001645e+02, 4.4251534e+01, 7.0184227e+01,
       2.6196262e+02, 4.6308620e+01, 5.0304062e+01, 3.4162041e+01,
       5.0358822e+01, 2.5223104e+01, 1.1894136e+02, 1.2442947e+02,
       1.8106448e+02, 2.5273548e+01, 2.0345103e+02, 8.1137140e+02,
       1.0323892e+03, 1.6790060e+02, 7.1853986e+02, 1.8901833e+02,
       1.4937018e+02, 2.1550936e+05, 2.1550936e+05, 4.5745233e+02,
       3.0354681e+02, 3.3761990e+02, 5.1620625e+03, 5.5263074e+02,
       3.7792105e+02, 1.5643800e+03, 1.1631915e+03, 5.4485138e+02,
       9.4224677e+02, 3.5535454e+03, 7.9113165e+02, 1.1008953e+03,
       6.8373911e+03, 6.9503418e+03, 2.9468035e+03, 9.2361578e+02,
       3.6889176e+04, 2.4262538e+02, 4.6831313e+03, 1.1353474e+04,
       3.7475058e+02, 2.0329285e+03, 5.8104077e+02, 7.7382671e+03,
       3.8380178e+03, 2.3634763e+03, 1.8499717e+04, 1.0784764e+03,
       1.2828820e+03, 1.5898770e+04, 2.8084089e+03, 3.9069910e+03,
       2.1550936e+05, 7.8065057e+02, 3.8844534e+03, 9.8346396e+03,
       2.9807143e+04, 1.4063845e+03, 2.6184104e+03, 2.1346461e+04,
       1.6832722e+03, 9.9057109e+03, 7.8587910e+03, 6.3752285e+03,
       3.7545466e+03, 3.7207004e+04, 8.0176318e+03, 1.0068148e+04,
       6.4197648e+04, 1.4779485e+04, 4.8537002e+03, 2.3645441e+04,
       5.1113359e+04, 2.6971604e+03, 8.8885654e+03, 4.4198578e+04,
       1.3318511e+04, 1.1935956e+04, 5.2379695e+04, 1.9229268e+04,
       4.4346191e+04, 2.6438607e+04, 4.2990035e+04, 2.5564508e+04,
       2.1550936e+05, 9.4947285e+03, 8.5185469e+03, 2.1550936e+05,
       3.5669277e+04, 9.3623506e+03, 2.5474133e+04, 1.2431808e+05,
       7.5106130e+00])).astype(np.float32)

epochs = 10000
# lr = 0.001
# batch_size = 50000
batch_size = 8

if(parallel):
    model_name = "ScanNet Finetuned SegFormer DICE - validation subsample multi-gpu {} - LR {:.2e} - {}".format(loss_weights,lr,gpu)
else:
    model_name = "ScanNet All Finetuned SegFormer DICE - validation subsample {} LR {:.2e} - {}".format(loss_weights,lr,gpu)

model_name = model_name.replace('.','_')

hub_model_id = model_name

if(not parallel):
    training_args = TrainingArguments(
        model_name,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=400,
        eval_steps=400,
        logging_steps=1,
        eval_accumulation_steps=1,
        gradient_accumulation_steps = 3,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        dataloader_num_workers = 8,
        batch_eval_metrics = True,
        fp16 = True,
        dataloader_pin_memory=True,
        # lr_scheduler_type = 'reduce_lr_on_plateau',
    )
else:
    training_args = TrainingArguments(
    model_name,
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=100,
    eval_steps=100,
    logging_steps=1,
    eval_accumulation_steps=1,
    gradient_accumulation_steps = 2,
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    dataloader_num_workers = 8,
    batch_eval_metrics = True,
    fp16 = True,
    local_rank=-1,  # This will be automatically set when launching with torchrun
    ddp_backend="nccl",
    dataloader_pin_memory=True,
    )


# metric = evaluate.load("mean_iou")

class MetricComputer:
    def __init__(self,n_classes = 101):
        self.n_classes = n_classes
        self.miou_calc = Cumulative_mIoU_torch(n_classes = self.n_classes)

    def compute_metrics(self,eval_pred,compute_result):
        with torch.no_grad():
            
            logits_tensor, labels = eval_pred
            # pdb.set_trace()
            # labels[labels == 255] = N_CLASSES-1

            # logits_tensor = torch.from_numpy(logits)
    #         print(logits_tensor.size())
            # scale the logits to the size of the label
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)
            pred_labels = logits_tensor.detach()
            label = labels
            # currently using _compute instead of compute
            # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
            if(not compute_result):
                # metrics = metric.add_batch(
                #         predictions=pred_labels,
                #         references=labels,
                #     )
                self.miou_calc.update_counts(pred_labels.flatten(),gt = label.flatten())
                return {}
            else:
                metrics = {}
                per_category_iou = self.miou_calc.get_IoUs()
                metrics.update({f"iou_{i}": v for i, v in enumerate(per_category_iou)})

                per_category_iou[np.logical_not(np.isfinite(per_category_iou))] = 0
                metrics.update({'mean_iou':np.mean(per_category_iou)})

                per_category_recall = self.miou_calc.get_recall()
                metrics.update({f"recall_{i}": v for i, v in enumerate(per_category_recall)})

                per_category_precision = self.miou_calc.get_precision()
                metrics.update({f"precision_{i}": v for i, v in enumerate(per_category_precision)})
                self.miou_calc.reset_counts()

            return metrics
    
early_stop = EarlyStoppingCallback(50,0.0005)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch = 10):
        labels = inputs.get("labels").to(model.device)
        # labels[labels == 255] = N_CLASSES-1
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits').to(model.device)
        logits = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).contiguous()
        # compute custom loss
        # pdb.set_trace()

        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(model.device))
        # pdb.set_trace()
        loss = loss_fct(logits, labels) + multiclass_dice_loss(logits,labels)
        return (loss, outputs) if return_outputs else loss

metrics_calculator = MetricComputer(n_classes = N_CLASSES)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=metrics_calculator.compute_metrics,
    callbacks=[early_stop]
)


trainer.train()

#saving the trained model to the best_model directory
trainer.save_model("./segmentation_model_checkpoints/{}".format(model_name))


