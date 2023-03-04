import cv2
import numpy as np
import os

img = cv2.imread("./frame.png")
height, width = img.shape[:2]
imgL = img[:, :int(width/2)]
imgR = img[:, int(width/2):]
os.makedirs("demo", exist_ok=True)
cv2.imwrite("demo/l.png", imgL)
cv2.imwrite("demo/r.png", imgR)
