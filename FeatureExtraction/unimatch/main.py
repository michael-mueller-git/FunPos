import cv2
import numpy as np
from app import inference

cap = cv2.VideoCapture('demo/test.mkv')
prev = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    img = cv2.resize(frame, None, fx=0.2, fy=0.2)
    height, width = img.shape[:2]
    imgL = img[:, :int(width/2)]
    imgR = img[:, int(width/2):]
    result_stereo = np.array(inference(imgL, imgR, "stereo"))
    if prev is not None:
        result_flow = np.array(inference(prev, imgL, "flow"))
        cv2.imshow('flow', result_flow)

    cv2.imshow('stereo', result_stereo)
    if cv2.waitKey(1) == ord('q'):
        break

    prev = imgL


cv2.destroyAllWindows()
cap.release()
