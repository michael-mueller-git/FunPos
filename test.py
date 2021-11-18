import cv2
import torch
import config
import json
import einops

from lib.funscript import Funscript
from lib.ffmpegstream import FFmpegStream
from model.model import FunPosModel

import numpy as np



WEIGHTS='./checkpoint/FunPos_ep_014'
TEST_FILE='./data/test/example1.mkv'


if __name__ == '__main__':
    print('Load', WEIGHTS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FunPosModel().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device)['model_state_dict'])
    model.eval()

    cap = cv2.VideoCapture(TEST_FILE)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    funscript = Funscript(fps)
    cap.release()

    with open("".join(TEST_FILE[:-4]) + '.param', "r") as f:
        param = json.load(f)

    video = FFmpegStream(TEST_FILE, param, skip_frames=config.skip_frames)
    frames, frame_numer = [], 0
    for i in range(config.seq_len):
        image = video.read()
        frames.append(image - config.IMAGE_MEAN)
        frame_numer += (config.skip_frames  + 1)

    with torch.no_grad():
        while video.isOpen():
            image = video.read()
            del frames[0]
            frames.append(image - config.IMAGE_MEAN)
            frame_numer += (config.skip_frames  + 1)
            frames_array = einops.rearrange(np.array(frames), 'time height width channel -> time channel height width')
            frames_tensor = torch.from_numpy(frames_array).float().unsqueeze(0).to(device)
            x = model(frames_tensor)
            x = x.cpu().numpy()[0][0][0]
            x = min((1.0, max((0.0, x))))
            print('Frame:', frame_numer, 'value:', x)
            funscript.add_action(round(x * 100), round(frame_numer * (1000/fps)))

    funscript.save("".join(TEST_FILE)[:-4] + ".funscript")
    print("done")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
