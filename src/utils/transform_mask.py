import numpy as np
import cv2
import os

MASK_PATH = '../../dataset/masks/patches/LC08_L1GT_117060_20200825_20200825_01_RT_Kumar-Roy_p00625.tif'
OUTPUT_DIR = './output'
OUTPUT_NAME = 'mask.png'

mask = cv2.imread(MASK_PATH)
copy = cv2.imread(MASK_PATH)

height, width, depth = mask.shape

for i in range(0, height):
    for j in range(0, width):
        if (mask[i,j,0] > 0):
            copy[i,j] = 255


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

cv2.imwrite(os.path.join(OUTPUT_DIR, OUTPUT_NAME), copy)
