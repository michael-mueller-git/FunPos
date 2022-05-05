import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image

import matplotlib.pyplot as plt

WEIGHTS = './out/maskRCNN_2'
NUM_CLASSES = 2
HIDDEN_LAYER = 256
IMG = './data/train/0_9.png'

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
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model = model.double()
model.eval()
img = Image.open(IMG).convert("RGB")
transforms = [torchvision.transforms.ToTensor()]
for i in transforms:
    img = i(img)
img = img.double()
output = model([img.to(device)])

for i in range(1):
    output_im = output[i]['masks'][0][0, :, :].cpu().detach().numpy()
    for k in range(len(output[i]['masks'])):
        output_im2 = output[i]['masks'][k][0, :, :].cpu().detach().numpy()
        output_im2[output_im2>0.5] = 1
        output_im2[output_im2<0.5] = 0
        output_im = output_im+output_im2

    output_im[output_im>0.5] = 1
    output_im[output_im<0.5] = 0

    plt.imshow(output_im, cmap='Greys')
    plt.show()

