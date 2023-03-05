import torch
import os
import cv2
import contextlib

import numpy as np

from rembg import remove

# midas imports
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

from torchvision.transforms import Compose, transforms

def download_file(filename, url):
    print("Downloading", url, "to", filename)
    torch.hub.download_url_to_file(url, filename)
    if not os.path.exists(filename):
        raise RuntimeError('Download failed. Try again later or manually download the file to that location.')

def estimatemidas(device, img, model, w, h, resize_mode, normalization):
    # init transform
    transform = Compose(
        [
            Resize(
                w,
                h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    # transform input
    img_input = transform({"image": img})["image"]

    # compute
    precision_scope = torch.autocast if device == torch.device("cuda") else contextlib.nullcontext
    with torch.no_grad(), precision_scope("cuda"):
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        if device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


def run_model(model_type, background_removal=True, invert_depth = False):
    model_dir = "./models/midas"
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #"dpt_beit_large_512" midas 3.1
    if model_type == 1:
        model_path = f"{model_dir}/dpt_beit_large_512.pt"
        print(model_path)
        if not os.path.exists(model_path):
            download_file(model_path,"https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt")
        model = DPTDepthModel(
            path=model_path,
            backbone="beitl16_512",
            non_negative=True,
        )
        net_w, net_h = 512, 512
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    #"dpt_beit_large_384" midas 3.1
    if model_type == 2:
        model_path = f"{model_dir}/dpt_beit_large_384.pt"
        print(model_path)
        if not os.path.exists(model_path):
            download_file(model_path,"https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_384.pt")
        model = DPTDepthModel(
            path=model_path,
            backbone="beitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    #"dpt_large_384" midas 3.0
    if model_type == 3:
        model_path = f"{model_dir}/dpt_large-midas-2f21e586.pt"
        print(model_path)
        if not os.path.exists(model_path):
            download_file(model_path,"https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt")
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    #"dpt_hybrid_384" midas 3.0
    elif model_type == 4:
        model_path = f"{model_dir}/dpt_hybrid-midas-501f0c75.pt"
        print(model_path)
        if not os.path.exists(model_path):
            download_file(model_path,"https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt")
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    #"midas_v21"
    elif model_type == 5:
        model_path = f"{model_dir}/midas_v21-f6b98070.pt"
        print(model_path)
        if not os.path.exists(model_path):
            download_file(model_path,"https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt")
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    #"midas_v21_small"
    elif model_type == 6:
        model_path = f"{model_dir}/midas_v21_small-70d6b9c8.pt"
        print(model_path)
        if not os.path.exists(model_path):
            download_file(model_path,"https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt")
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        raise Exception("Invalid Model")

    # prepare for evaluation
    model.eval()

    # optimize
    if device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    print("Computing depthmap(s) ..")
    img_bgr = cv2.imread("./input.jpg")
    img_rgb = cv2.cvtColor(np.asarray(img_bgr), cv2.COLOR_BGR2RGB) / 255.0
    net_width, net_height = img_rgb.shape[:2]
    prediction = estimatemidas(device, img_rgb, model, net_width, net_height, resize_mode, normalization)
    depth = prediction
    numbytes=2
    depth_min = depth.min()
    depth_max = depth.max()
    max_val = (2**(8*numbytes))-1
    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape)

    img_output = out.astype("uint16")

    if invert_depth ^ model_type == 0:
        img_output = cv2.bitwise_not(img_output)

    if background_removal:
        background_removed_array = np.array(remove(img_bgr))
        bg_mask = (background_removed_array[:,:,0]==0)&(background_removed_array[:,:,1]==0)&(background_removed_array[:,:,2]==0)&(background_removed_array[:,:,3]<=0.2)
        far_value = 255 if invert_depth else 0
        img_output[bg_mask] = far_value * far_value

    img_output2 = np.zeros_like(img_rgb)
    img_output2[:,:,0] = img_output / 256.0
    img_output2[:,:,1] = img_output / 256.0
    img_output2[:,:,2] = img_output / 256.0

    cv2.imwrite("out.png", img_output2)

if __name__ == "__main__":
    run_model(6)
