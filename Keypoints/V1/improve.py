import os
import sys

IMAGES_DIR = './data/train/images'
ANNOTATION_DIR = './data/train/annotations'
DIGITS = 8

files = [f for f in os.listdir('./data/train/images')]

for f in files:
    num, _ = f.split('_')[-1].split('.', 2)
    num_int = int(num)
    searchA = os.path.join(IMAGES_DIR, f.replace(num, str(num_int-1).zfill(DIGITS)))
    searchB = os.path.join(IMAGES_DIR, f.replace(num, str(num_int+1).zfill(DIGITS)))
    if not os.path.exists(searchA):
        continue
    if not os.path.exists(searchB):
        continue

    annotation_file = os.path.join(ANNOTATION_DIR, f.replace('.jpg', '.json'))
    img_file = os.path.join(IMAGES_DIR, f)
    print("delete", annotation_file, img_file)
    os.remove(annotation_file)
    os.remove(img_file)
