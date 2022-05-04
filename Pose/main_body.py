from utils.ffmpegstream import FFmpegStream
from utils.vrprojection import VrProjection
from utils.ppca import PPCA
import cv2
import json
import copy
import os
import sys

import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt


class BodyMovePredictor:

    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.result = {
                11: [],
                12: [],
                23: [],
                24: []
            }

    def get_result(self):
        _, _, _, principalComponents, _ = PPCA(np.transpose(np.array([self.result[k] for k in self.result.keys()], dtype=float)), d=1)
        merged = [item[0] for item in principalComponents.tolist()]
        return merged


    def update(self, frame):
        self.img_height, self.img_width = frame.shape[:2]
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = self.pose.process(imgRGB)
        if not pose_result.pose_landmarks:
            for k in self.result.keys():
                self.result[k].append(None)
        else:
            for k in self.result.keys():
                self.result[k].append(pose_result.pose_landmarks.landmark[k].y if pose_result.pose_landmarks.landmark[k].visibility > 0.2 else None)

            for i in range(33):
                if i not in self.result.keys():
                    pose_result.pose_landmarks.landmark[i].visibility = 0.0
            self.mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return frame



video_file = './data/example1.mkv'

if not os.path.exists(video_file):
    print('file not found', video_file)
    sys.exit(1)

projection = VrProjection(video_file)
config = projection.get_parameter()
video = FFmpegStream(video_file, config)
frame = video.read()
bodyMovePredictor = BodyMovePredictor()

while video.isOpen():
    frame = video.read()
    if frame is None: break
    frame = bodyMovePredictor.update(frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.stop()
cv2.destroyAllWindows()

result = bodyMovePredictor.get_result()

plt.plot(result)
plt.show()
