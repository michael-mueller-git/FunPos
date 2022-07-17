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


class BlowjobPredictor:

    def __init__(self, dick_bbox):
        self.dick_bbox = dick_bbox
        self.face = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence = 0.5,
                min_tracking_confidence = 0.6)
        self.hands = mp.solutions.hands.Hands(
                model_complexity = 0,
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.result = {
                'mouth': [],
                'handLeft': [],
                'handRight': []
                }

    def is_touch_dick(self, x, y):
        if x < self.dick_bbox[0]:
            return False
        if y < self.dick_bbox[1]:
            return False
        if x > self.dick_bbox[0] + self.dick_bbox[2]:
            return False
        if y > self.dick_bbox[1] + self.dick_bbox[3]:
            return False

        return True


    def get_result(self):
        return self.result


    def update(self, frame):
        # face_keypoint_idx = [11,12,13,14,15,16,17]
        face_keypoint_idx = [13]
        hand_keypoint_idx = [5, 9, 13, 17]
        self.img_height, self.img_width = frame.shape[:2]
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = self.face.process(imgRGB)
        hands_result = self.hands.process(imgRGB)
        cv2.rectangle(frame,
                (self.dick_bbox[0], self.dick_bbox[1]),
                ((self.dick_bbox[0] + self.dick_bbox[2]), (self.dick_bbox[1] + self.dick_bbox[3])),
                (255,255,255), 3, 1)
        if face_result.multi_face_landmarks:
            face = face_result.multi_face_landmarks[0].landmark
            x, y = [], []
            for idx in face_keypoint_idx:
                x.append(round(face[idx].x * self.img_width))
                y.append(round(face[idx].y * self.img_height))
            x = round(sum(x) / len(x))
            y = round(sum(y) / len(y))
            cv2.circle(frame, (x, y), 4, (255, 0, 255), 2)
            if self.is_touch_dick(x, y):
                y /= self.img_height
                self.result['mouth'].append(y)
            else:
                None
        else:
            self.result['mouth'].append(None)

        if hands_result.multi_hand_landmarks:
            left = None
            right = None
            for hand_labels, hand_landmarks in zip(hands_result.multi_handedness, hands_result.multi_hand_landmarks):
                # self.mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                hand = hand_landmarks.landmark
                hand_type = hand_labels.classification[0].label
                x, y = [], []
                for idx in hand_keypoint_idx:
                    x.append(round(hand[idx].x * self.img_width))
                    y.append(round(hand[idx].y * self.img_height))
                x = round(sum(x) / len(x))
                y = round(sum(y) / len(y))
                cv2.circle(frame, (x, y), 4, (255, 255, 0), 2)
                if self.is_touch_dick(x, y):
                    y /= self.img_height
                    if hand_type == 'Right':
                        right = y
                    elif hand_type == 'Left':
                        left = y
                    else:
                        print('invalid hand type', hand_type)

            self.result['handLeft'].append(left)
            self.result['handRight'].append(right)
        else:
            for x in ['Right', 'Left']:
                self.result['hand'+str(x)].append(None)

        return frame


video_file = './data/example6.mp4'

if not os.path.exists(video_file):
    print('file not found', video_file)
    sys.exit(1)

projection = VrProjection(video_file)
config = projection.get_parameter()
video = FFmpegStream(video_file, config)
frame = video.read()
dick_bbox = cv2.selectROI("select dick", frame)
blowjobPredictor = BlowjobPredictor(dick_bbox)

while video.isOpen():
    frame = video.read()
    if frame is None: break
    frame = blowjobPredictor.update(frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.stop()
cv2.destroyAllWindows()

result = blowjobPredictor.get_result()

plt.plot(result['mouth'])
plt.plot(result['handRight'])
plt.plot(result['handLeft'])
plt.show()
