from utils.ffmpegstream import FFmpegStream
from utils.vrprojection import VrProjection
from utils.ppca import PPCA
import cv2
import json
import copy
import os
import sys

import numpy as np

SKIP_FRAMES = 3
video_file = './data/raw/example6.mp4'
out_path = './data/export'

if not os.path.exists(video_file):
    print('file not found', video_file)
    sys.exit(1)

if not os.path.exists(out_path):
    os.mkdir(out_path)

projection = VrProjection(video_file)
config = projection.get_parameter()
video = FFmpegStream(video_file, config)
frame = video.read()

frame_num = 0
while video.isOpen():
    frame = video.read()

    if frame is None: break
    frame_num += 1

    if frame_num % SKIP_FRAMES != 0:
        continue

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(10)

    if key == ord('q'):
        break

    name = '{}/{}_{}'.format(out_path, os.path.basename(video_file), str(frame_num).zfill(6))
    print('save', name)
    cv2.imwrite(name + '.png', frame)


video.stop()
cv2.destroyAllWindows()

