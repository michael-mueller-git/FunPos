from utils.ffmpegstream import FFmpegStream
from utils.vrprojection import VrProjection
import cv2
import os
import sys
import torch
from model import PoseModel, DetModel

import numpy as np

SCALE = 0.5
detector_name = 'YOLOX-l'
pose_model_name = 'ViTPose-L (single-task train)'
det_score_threshold = 0.5
max_num_frames = 300
vis_kpt_score_threshold = 0.3
vis_dot_radius = 4
vis_line_thickness = 2
video_file = './data/example4.mkv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(video_file):
    print('file not found', video_file)
    sys.exit(1)

projection = VrProjection(video_file)
config = projection.get_parameter()
video = FFmpegStream(video_file, config)

det_model = DetModel(device, detector_name)
pose_model = PoseModel(device, pose_model_name)
cv2.destroyAllWindows()

i = 0
init_bbox = None
while video.isOpen():
    img = video.read()
    if img is None: break
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE)
    rgb_frame = img[:, :, ::-1]
    det_preds = det_model.detect(rgb_frame)
    preds, vis = pose_model.predict_pose_and_visualize(
        rgb_frame, det_preds, det_score_threshold, vis_kpt_score_threshold,
        vis_dot_radius, vis_line_thickness)

    i += 1
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imshow('Frame', vis)
    if cv2.waitKey(1) == ord('q'):
        break

video.stop()
cv2.destroyAllWindows()

