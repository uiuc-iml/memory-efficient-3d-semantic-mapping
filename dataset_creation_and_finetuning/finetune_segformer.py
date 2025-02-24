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


from datasets import IterableDataset,Dataset

# def data_generator():
#     i = 0
#     semantic_frames = pd.Series(sorted(glob('/home/motion/data/scannet_pp/data/**/*.png',recursive = True)))
#     frame_names = semantic_frames.str.split('/',expand = True).iloc[:,-1].str.split('.',expand = True).iloc[:,0]
#     rgb_frames = semantic_frames.str.rsplit('/',expand = True,n = 2).iloc[:,0] + '/rgb/'+frame_names+'.jpg'
#     n = rgb_frames.shape[0]
#     while(i<n):
#         try:
#             tmp = cv2.imread(semantic_frames[i],cv2.IMREAD_UNCHANGED)
#             tmp2 = cv2.imread(rgb_frames[i])
#             i +=1
#             yield {'rgb':tmp2,'gt':tmp}
#         except Exception as e:
#             i+=1
#             continue
N_CLASSES = 101

# ds = IterableDataset.from_generator(data_generator)


def multiclass_dice_loss(pred, target, smooth=1):
    """
    Computes Dice Loss for multi-class segmentation. Thanks to https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
    Args:
        pred: Tensor of predictions (batch_size, C, H, W).
        target: One-hot encoded ground truth (batch_size, C, H, W).
        smooth: Smoothing factor.
    Returns:
        Scalar Dice Loss.

    """
    pred = F.softmax(pred, dim=1)  # Convert logits to probabilities
    num_classes = pred.shape[1]  # Number of classes (C)
    dice = 0  # Initialize Dice loss accumulator
    
    for c in range(num_classes):  # Loop through each class
        pred_c = pred[:, c]  # Predictions for class c
        target_c = target[:, c]  # Ground truth for class c
        
        intersection = (pred_c * target_c).sum(dim=(1, 2))  # Element-wise multiplication
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))  # Sum of all pixels
        
        dice += (2. * intersection + smooth) / (union + smooth)  # Per-class Dice score

    return 1 - dice.mean() / num_classes  # Average Dice Loss across classes



def load_images(rgb,semantic,):
    # import pdb
    # pdb.set_trace()
    tmp = cv2.imread(semantic[0],cv2.IMREAD_UNCHANGED).astype(np.uint8)
    tmp2 = cv2.imread(rgb[0]).astype(np.uint8)
    tmp2 = np.moveaxis(tmp2,[0,1,2],[1,2,0])
    return tmp2,tmp


parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


# fnames = get_filenames()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model =  SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512").to(device)


model.train()
for param in model.parameters():
    param.requires_grad = False
decoder_head = model.decode_head
decoder_head.classifier = nn.Conv2d(768,N_CLASSES,kernel_size=(1, 1), stride=(1, 1))
for param in decoder_head.parameters():
    param.requires_grad = True

model.to(device)
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
val_ds = Dataset.load_from_disk(val_ds_dir)

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

# inverse class frequency error weighing
# weights = np.array(([ 0.86863202,  1.        ,  1.26482577,  4.97661045,  6.21128435,
#         4.0068586 ,  8.72477767,  4.93037224,  5.65326448, 16.44580194,
#        18.8649601 , 55.24242013, 29.60985561, 11.04643569, 20.82360894,
#        30.38149462, 45.75461865, 32.74486949, 50.70553433, 30.23161118,
#         7.48407616])).astype(np.float32)

# weights = np.ones(151)
# weights = np.array([3.57321040e+00, 0.00000000e+00, 0.00000000e+00, 7.08178666e+00,
#        0.00000000e+00, 4.14079262e+01, 0.00000000e+00, 2.05134159e+04,
#        5.27260114e+01, 0.00000000e+00, 2.07407545e+01, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 4.02161544e+01, 1.14332567e+01,
#        0.00000000e+00, 3.52378877e+02, 2.70503919e+02, 2.02602728e+01,
#        0.00000000e+00, 0.00000000e+00, 6.34202531e+02, 1.95732787e+02,
#        1.67273495e+02, 0.00000000e+00, 0.00000000e+00, 4.64582446e+03,
#        8.40246160e+03, 0.00000000e+00, 1.34579615e+03, 6.83789516e+03,
#        0.00000000e+00, 1.43370158e+02, 0.00000000e+00, 1.06374526e+03,
#        3.27712097e+02, 0.00000000e+00, 4.72086901e+05, 2.20186432e+03,
#        0.00000000e+00, 8.57574874e+01, 5.25179543e+02, 0.00000000e+00,
#        6.46439060e+05, 6.31165003e+02, 0.00000000e+00, 1.17806968e+03,
#        0.00000000e+00, 0.00000000e+00, 9.88847656e+02, 0.00000000e+00,
#        0.00000000e+00, 1.14776052e+03, 0.00000000e+00, 2.04510285e+03,
#        0.00000000e+00, 8.25563342e+03, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 3.74999491e+01, 7.04992439e+01,
#        0.00000000e+00, 1.02464345e+04, 3.56803158e+04, 2.81996793e+02,
#        0.00000000e+00, 6.26742915e+02, 3.43953330e+04, 6.08058623e+03,
#        0.00000000e+00, 2.37217990e+02, 1.19177611e+02, 0.00000000e+00,
#        0.00000000e+00, 2.15204603e+05, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 6.61436138e+03, 1.23352336e+03, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        4.19741084e+02, 0.00000000e+00, 0.00000000e+00, 3.54751507e+04,
#        0.00000000e+00, 5.43585072e+04, 1.16217891e+03, 0.00000000e+00,
#        4.23087875e+02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.16356597e+03,
#        0.00000000e+00, 0.00000000e+00, 6.23274815e+02, 1.96132662e+04,
#        3.96258422e+03, 0.00000000e+00, 0.00000000e+00, 4.12327557e+02,
#        0.00000000e+00, 0.00000000e+00, 5.83507259e+03, 1.90547152e+04,
#        6.72617183e+04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        3.21199779e+03, 1.82823569e+03, 0.00000000e+00, 1.83371163e+04,
#        0.00000000e+00, 1.93259397e+03, 4.75639498e+02, 6.65932929e+03,
#        0.00000000e+00, 5.23085590e+02, 0.00000000e+00, 8.23645939e+03,
#        0.00000000e+00, 1.72973788e+03, 4.13863435e+02, 8.00092090e+02,
#        0.00000000e+00, 0.00000000e+00, 7.95932510e+03, 2.80106544e+01,
#        2.58610958e+01, 0.00000000e+00, 1.64592010e+02, 4.26831910e+03,
#        1.67464229e+04, 0.00000000e+00, 8.73879480e+00]).astype(np.float32)
# weights = np.ones(101).astype(np.float32)

# weights = np.array([3.59293393e+00, 4.38779666e+01, 6.91258655e+00, 1.04030130e+01,
#        4.34120532e+01, 3.50016457e+02, 4.42515333e+01, 7.01842234e+01,
#        2.61962616e+02, 4.63086187e+01, 5.03040602e+01, 3.41620424e+01,
#        5.03588200e+01, 2.52231054e+01, 1.18941359e+02, 1.24429469e+02,
#        1.81064488e+02, 2.52735480e+01, 2.03451028e+02, 8.11371405e+02,
#        1.03238920e+03, 1.67900605e+02, 7.18539883e+02, 1.89018332e+02,
#        1.49370179e+02, 0.00000000e+00, 0.00000000e+00, 4.57452338e+02,
#        3.03546824e+02, 3.37619909e+02, 5.16206244e+03, 5.52630734e+02,
#        3.77921064e+02, 1.56437996e+03, 1.16319155e+03, 5.44851377e+02,
#        9.42246742e+02, 3.55354549e+03, 7.91131652e+02, 1.10089526e+03,
#        6.83739112e+03, 6.95034156e+03, 2.94680346e+03, 9.23615814e+02,
#        3.68891747e+04, 2.42625389e+02, 4.68313131e+03, 1.13534741e+04,
#        3.74750572e+02, 2.03292852e+03, 5.81040767e+02, 7.73826721e+03,
#        3.83801779e+03, 2.36347641e+03, 1.84997171e+04, 1.07847647e+03,
#        1.28288199e+03, 1.58987696e+04, 2.80840903e+03, 3.90699108e+03,
#        0.00000000e+00, 7.80650546e+02, 3.88445326e+03, 9.83463922e+03,
#        2.98071433e+04, 1.40638455e+03, 2.61841032e+03, 2.13464609e+04,
#        1.68327223e+03, 9.90571137e+03, 7.85879103e+03, 6.37522840e+03,
#        3.75454666e+03, 3.72070025e+04, 8.01763194e+03, 1.00681482e+04,
#        6.41976474e+04, 1.47794855e+04, 4.85370019e+03, 2.36454421e+04,
#        5.11133612e+04, 2.69716039e+03, 8.88856575e+03, 4.41985790e+04,
#        1.33185103e+04, 1.19359562e+04, 5.23796968e+04, 1.92292683e+04,
#        4.43461910e+04, 2.64386066e+04, 4.29900354e+04, 2.55645085e+04,
#        2.15509367e+05, 9.49472882e+03, 8.51854705e+03, 0.00000000e+00,
#        3.56692790e+04, 9.36235036e+03, 2.54741333e+04, 1.24318078e+05,
#        7.51061301e+00]).astype(np.float32)

# set of weights with laplace smoothing
weights = np.sqrt(np.array([3.5929339e+00, 4.3877968e+01, 6.9125867e+00, 1.0403013e+01,
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
lr = 0.0001
batch_size = 500

hub_model_id = "finetuned ScanNetpp"

training_args = TrainingArguments(
    "ScanNet Finetuned SegFormer - with soft-DICE",
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=200,
    eval_steps=200,
    logging_steps=1,
    eval_accumulation_steps=1,
    gradient_accumulation_steps = 1,
    load_best_model_at_end=True,
    metric_for_best_model='mean_iou',
    dataloader_num_workers = 8,
    fp16 = False
    # lr_scheduler_type = 'reduce_lr_on_plateau',
)

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):

    start = time.time()
    with torch.no_grad():
        
        logits, labels = eval_pred
        # labels[labels == 255] = N_CLASSES-1

        logits_tensor = torch.from_numpy(logits)
#         print(logits_tensor.size())
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        label = labels
        # currently using _compute instead of compute
        # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
        metrics = metric._compute(
                predictions=pred_labels,
                references=labels,
                num_labels=N_CLASSES,
                ignore_index=False,
                reduce_labels=feature_extractor.do_reduce_labels,
            )

        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"accuracy_{i}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{i}": v for i, v in enumerate(per_category_iou)})
        print('metric calculations took {}'.format(time.time()-start))
        return metrics

early_stop = EarlyStoppingCallback(150,0.0005)


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
        )
        # compute custom loss
        # pdb.set_trace()

        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(model.device))
        # pdb.set_trace()
        loss = loss_fct(logits, labels) + multiclass_dice_loss(logits,labels)
        return (loss, outputs) if return_outputs else loss
    
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[early_stop]
)


trainer.train()

#saving the trained model to the best_model directory
trainer.save_model("./segmentation_model_checkpoints/Segformer")


