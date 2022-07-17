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

MODEL_COMPLEXITY = 0 # Can be 0, 1, or 2. If higher complexity is chosen, the inference time increases.
DRAW_LANDMARKS = False

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

video_file = './data/example2.mkv'

if not os.path.exists(video_file):
    print('file not found', video_file)
    sys.exit(1)

projection = VrProjection(video_file)
config = projection.get_parameter()
video = FFmpegStream(video_file, config)

points_y = {
        11: [],
        12: [],
        23: [],
        24: []
}

with mp_holistic.Holistic(
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        model_complexity=MODEL_COMPLEXITY,
        static_image_mode=False # True: person detection runs for every input image. False: Detection runs once followed by landmark tracking
        ) as holistic:
    while video.isOpen():
        frame = video.read()
        if frame is None: break
        image = frame
        results = holistic.process(image)

        for k in points_y.keys():
            points_y[k].append(results.pose_landmarks.landmark[k].y if results.pose_landmarks.landmark[k].visibility > 0.2 else None)

        if DRAW_LANDMARKS:
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

video.stop()
cv2.destroyAllWindows()


# for k in points_y.keys():
#     plt.plot(points_y[k])

# plt.plot()


_, _, _, principalComponents, _ = PPCA(np.transpose(np.array([points_y[k] for k in points_y.keys()], dtype=float)), d=1)
merged = [item[0] for item in principalComponents.tolist()]
# print('merged', merged)


plt.plot(merged)
plt.show()
