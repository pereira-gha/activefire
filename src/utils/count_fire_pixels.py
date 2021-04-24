"""Count the amout of fire pixels in the image
If the path is informed only it will be computed, otherwise all patches of the image will be computed
"""

import os
import sys
from glob import glob
import numpy as np
import rasterio 


MASK_NAME = 'LC08_L1TP_003006_20200911_20200911_01_RT_Voting.TIF'
#IMAGE_NAME = 'LC08_L1GT_117060_20200825_20200825_01_RT_Kumar-Roy'

MASK_PATH = '../../dataset/manual_annotations/scenes/masks/'
PATCHES_PATTERN = '*.tif'

def get_mask_arr(path):
    """ Abre a mascara como array"""
    with rasterio.open(path) as src:
        img = src.read().transpose((1, 2, 0))
        seg = np.array(img, dtype=int)

        return seg[:, :, 0]


if __name__ == '__main__':

    image_name = MASK_NAME
    if not MASK_NAME.endswith('.tif') and not MASK_NAME.endswith('.TIF'):
        image_name = MASK_NAME + PATCHES_PATTERN

    images_path = glob(os.path.join(MASK_PATH, image_name))
    print('Mask: {}'.format(image_name))
    print('Total images found: {}'.format( len(images_path) ))
    count = 0
    for image_path in images_path:
        mask = get_mask_arr(image_path)

        count += (mask > 0).sum()

    print('Total Fire Pixels: {}'.format(count))
