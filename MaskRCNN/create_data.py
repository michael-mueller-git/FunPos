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
VIDEO_NUMBER=0
video_file = './data/raw/example1.mkv'
out_path = './data/train'

if not os.path.exists(video_file):
    print('file not found', video_file)
    sys.exit(1)

projection = VrProjection(video_file)
config = projection.get_parameter()
video = FFmpegStream(video_file, config)
frame = video.read()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
frame_num = 0
while video.isOpen():
    frame = video.read()
    if frame is None: break
    frame_num += 1
    if frame_num % SKIP_FRAMES != 0:
        continue

    roi = cv2.selectROI("Frame", frame)

    mask = np.zeros(frame.shape[:2], np.uint8)
    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(frame, mask, roi, backgroundModel, foregroundModel, 3, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    edit_mask = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask3 = cv2.bitwise_or(mask2, edit_mask)

    mask_out = mask3
    mask_image = frame * mask_out[:, :, np.newaxis]
    cv2.imshow('Frame', mask_image)
    # cv2.imshow('Frame', mask_out*255)
    key = cv2.waitKey()

    if key == ord('q'):
        break
    elif key == ord(' '):
        if cv2.countNonZero(mask_out) > 5:
            name = '{}/{}_{}'.format(out_path, VIDEO_NUMBER, frame_num)
            print('save', name)
            cv2.imwrite(name + '.png', frame)
            cv2.imwrite(name + '_anno.png', mask_out * 255)
        else:
            print('skip does not contain a mask')
    # every other key = skip this mask


video.stop()
cv2.destroyAllWindows()

