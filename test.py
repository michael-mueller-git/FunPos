import cv2
import torch
import sys
import os
import json
import einops

from utils.funscript import Funscript
from utils.ffmpegstream import FFmpegStream
from utils.config import CONFIG
from model.model1 import FunPosModel
from model.model2 import FunPosTransformerModel

import numpy as np


MODEL = CONFIG['general']['select']
MODEL_CLASS = CONFIG[MODEL]['class']
SEQ_LEN = CONFIG[MODEL]['seq_len']
SKIP_FRAMES = CONFIG[MODEL]['skip_frames']
TEST_FILE = CONFIG['general']['test_file']
CHEKPOINT_DIR = CONFIG['general']['checkpoint_dir']

def get_weights_file():
    if not os.path.exists(CHEKPOINT_DIR):
        print("checkpoint directory does not exist")
        sys.exit()
    cp = [os.path.join(CHEKPOINT_DIR, f) for f in os.listdir(CHEKPOINT_DIR) if f.startswith(CONFIG[MODEL]['name'] + '_ep')]
    if len(cp) < 1:
        print("checkpoint for selected model not available")
        sys.exit()
    cp.sort()
    return cp[-1]

WEIGHTS = get_weights_file()

if __name__ == '__main__':
    print('Load', WEIGHTS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("create model", MODEL_CLASS)
    exec('model = ' + MODEL_CLASS + '().to(device)')
    model.load_state_dict(torch.load(WEIGHTS, map_location=device)['model_state_dict'])
    model.eval()

    cap = cv2.VideoCapture(TEST_FILE)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    funscript = Funscript(fps)
    cap.release()

    with open("".join(TEST_FILE[:-4]) + '.param', "r") as f:
        param = json.load(f)

    video = FFmpegStream(TEST_FILE, param, skip_frames=SKIP_FRAMES)
    frames, frame_numer = [], 0
    for i in range(SEQ_LEN):
        image = video.read()
        frames.append(image)
        frame_numer += (SKIP_FRAMES  + 1)

    with torch.no_grad():
        while video.isOpen():
            image = video.read()
            del frames[0]
            frames.append(image)
            frame_numer += (SKIP_FRAMES  + 1)
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
