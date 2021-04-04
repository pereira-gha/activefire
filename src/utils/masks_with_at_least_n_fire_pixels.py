"""This scripts count the white pixels in a mask and if it has more pixels than NUM_PIXELS the mask path is printed.
It can be use to find images and/or masks with a minimal amount of fire pixels.
"""

import os
import sys
from glob import glob
import numpy as np
import rasterio 

MASK_PATH = '../../dataset/groundtruth/manual_annotation'
NUM_PIXELS = 500

def get_mask_arr(path):
    """ Abre a mascara como array"""
    with rasterio.open(path) as src:
        img = src.read().transpose((1, 2, 0))
        seg = np.array(img, dtype=int)

        return seg[:, :, 0]


if __name__ == '__main__':


    images_path = glob(os.path.join(MASK_PATH, '*.tif'))
    print('Mask Path: {}'.format(MASK_PATH))
    print('Total images found: {}'.format( len(images_path) ))
    num_images = 0
    for image_path in images_path:
        mask = get_mask_arr(image_path)

        count = (mask > 0).sum()

        if count > NUM_PIXELS:
            print('# Fire Pixels: {} - Image: {}'.format(count, image_path))
            num_images += 1

    print('Num. images: {}'.format(num_images))
