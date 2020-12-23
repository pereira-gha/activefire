import rasterio
import numpy as np
import os
import cv2

IMAGE = '../../dataset/images/patches/LC08_L1GT_117060_20200825_20200825_01_RT_p00625.tif'
OUTPUT_DIR = './output'
OUTPUT_IMAGE_NAME = '762.png'

MAX_PIXEL_VALUE = 65535 # Max. pixel value, used to normalize the image

def get_img_762bands(path):
    img = rasterio.open(path).read((7,6,2)).transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img
        

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

img = get_img_762bands(IMAGE)

img = np.array(img * 255, dtype=np.uint8)
cv2.imwrite(os.path.join(OUTPUT_DIR, OUTPUT_IMAGE_NAME), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))