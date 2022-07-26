from utils.ffmpegstream import FFmpegStream
from utils.vrprojection import VrProjection
import cv2
import os
import sys
import mmcv
import torch, torchvision
import mmdet
import mmtrack
from mmtrack.apis import inference_sot, init_model

import numpy as np

SCALE = 1

video_file = './data/example4.mkv'

if not os.path.exists(video_file):
    print('file not found', video_file)
    sys.exit(1)

projection = VrProjection(video_file)
config = projection.get_parameter()
video = FFmpegStream(video_file, config)

sot_config = './mmtracking/configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py'
sot_checkpoint = './checkpoints/siamese_rpn_r50_1x_lasot_20211203_151612-da4b3c66.pth'
sot_model = init_model(sot_config, sot_checkpoint, device='cpu')

i = 0
init_bbox = None
while video.isOpen():
    img = video.read()
    if img is None: break
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE)
    if init_bbox is None:
        init_bbox = cv2.selectROI("Frame", img)
        print('init_bbox', init_bbox)
    result = inference_sot(sot_model, img, init_bbox, frame_id=i)
    # print(result)
    result['track_bboxes'] = np.array([int(result['track_bboxes'][0]), int(result['track_bboxes'][1]), int(result['track_bboxes'][2]), int(result['track_bboxes'][3]), result['track_bboxes'][4]])
    print(result)
    img = sot_model.show_result(img, result)
    i += 1
    cv2.imshow('Frame', img)
    if cv2.waitKey(1) == ord('q'):
        break

video.stop()
cv2.destroyAllWindows()

