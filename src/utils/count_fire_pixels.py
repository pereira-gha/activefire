"""Count the amout of fire pixels in the image
If the path is informed only it will be computed, otherwise all patches of the image will be computed
"""

import os
import sys
from glob import glob
import numpy as np
import rasterio 


IMAGE_NAME = 'LC08_L1TP_046031_20200908_20200908_01_RT'
IMAGE_PATH = '/home/andre/GROUNDTRUTH/GROUNDTRUTH_GABRIEL_patches/'
PATCHES_PATTERN = '*_v1_*.tif'

def get_mask_arr(path):
    """ Abre a mascara como array"""
    with rasterio.open(path) as src:
        img = src.read().transpose((1, 2, 0))
        seg = np.array(img, dtype=int)

        return seg[:, :, 0]


if __name__ == '__main__':

    image_name = IMAGE_NAME
    if not IMAGE_NAME.endswith('.tif'):
        image_name = IMAGE_NAME + PATCHES_PATTERN

    images_path = glob(os.path.join(IMAGE_PATH, image_name))
    print('Image: {}'.format(image_name))
    print('Total imagens found: {}'.format( len(images_path) ))
    count = 0
    for image_path in images_path:
        mask = get_mask_arr(image_path)

        count += (mask > 0).sum()

    print('Total Fire Pixels: {}'.format(count))