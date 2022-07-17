from dataset import SingleMaskDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

import os
import torch
import torchvision

import numpy as np


TRAIN_DATA_DIR = './data/train'
NUM_CLASSES = 2 # Number of classes (including background)
HIDDEN_LAYER = 256
N_EPOCHS = 20
SAVE_DIR = './out'

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

dataset_train = SingleMaskDataset(TRAIN_DATA_DIR)
data_loader_train = torch.utils.data.DataLoader(dataset_train,
        batch_size = 2,
        shuffle = True,
        collate_fn = lambda x:list(zip(*x))
    )

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features,
        NUM_CLASSES
    )

model.roi_heads.mask_predictor = MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels,
        HIDDEN_LAYER,
        NUM_CLASSES
    )

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model=model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

loss_list = []
model.train()

for epoch in tqdm(range(N_EPOCHS)):
    loss_epoch = []
    for (images, targets) in tqdm(data_loader_train):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        model=model.double()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        loss_epoch.append(losses.item())

    loss_epoch_mean = np.mean(loss_epoch)
    loss_list.append(loss_epoch_mean)
    print("Average loss for epoch {} = {:.4f} ".format(epoch, loss_epoch_mean))
    torch.save(model.state_dict(), "{}/MaskRCNN_{}".format(SAVE_DIR, epoch))

if torch.cuda.is_available():
    torch.cuda.empty_cache()
