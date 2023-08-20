import os
import torch
import sys

from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "lib", "keypoint_rcnn_training"))

from utils import collate_fn
from engine import train_one_epoch

from dataset import ClassDataset, train_transform
from model import get_model


def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    KEYPOINTS_FOLDER_TRAIN = 'data/train'
    dataset_train = ClassDataset(KEYPOINTS_FOLDER_TRAIN, transform=train_transform(), demo=False)
    data_loader_train = DataLoader(dataset_train, batch_size=3, shuffle=True, collate_fn=collate_fn)

    model = get_model(device, num_keypoints = 2)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    num_epochs = 5

    for epoch in range(num_epochs):
        print("epoch", epoch)
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=100)
        lr_scheduler.step()
        torch.save(model.state_dict(), f"keypointsrcnn_weights_{epoch}.pth")


if __name__ == "__main__":
    train()


