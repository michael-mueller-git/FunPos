import cv2
import torch
import config
import json
from lib.funscript import Funscript

from lib.ffmpegstream import FFmpegStream

import numpy as np

from model.model import FunPos_Model

WEIGHTS='./checkpoint/FunPos_ep_009'
TEST_FILE='./data/test/example1.mkv'

if __name__ == '__main__':
    print('Load', WEIGHTS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FunPos_Model().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device)['model_state_dict'])
    model.eval()

    cap = cv2.VideoCapture(TEST_FILE)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    funscript = Funscript(fps)
    cap.release()

    with open("".join(TEST_FILE[:-4]) + '.param', "r") as f:
        param = json.load(f)

    cap = FFmpegStream(TEST_FILE, param, skip_frames=config.skip_frames)
    frames = []
    frame_numer = 0
    for i in range(config.seq_len):
        image = cap.read()
        frames.append(image - config.IMAGE_MEAN)
        frame_numer += (config.skip_frames  + 1)

    with torch.no_grad():
        while cap.isOpen():
            image = cap.read()
            del frames[0]
            frames.append(image - config.IMAGE_MEAN)
            frame_numer += (config.skip_frames  + 1)
            frames_tensor = torch.from_numpy(np.array(frames).transpose([0, 3, 1, 2])).float().unsqueeze(0).to(device) # 1, time, channel, height, width
            x = model(frames_tensor)
            x = x.cpu().numpy()[0][0][0]
            x = min((1.0, max((0.0, x))))
            print('Frame:', frame_numer, 'value:', x)
            funscript.add_action(round(x * 100), round(frame_numer * (1000/fps)))

    funscript.save("".join(TEST_FILE)[:-4] + ".funscript")
    print("done")
    if torch.cuda.is_available(): torch.cuda.empty_cache()
