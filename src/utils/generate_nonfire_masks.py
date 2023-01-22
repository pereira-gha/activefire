"""
This script generates non-fire masks (zeros only) to the given patches.
Most of the images don't have fire on it, in order to save some space, this masks are not available in the dataset.
A non-fire mask can be (and are in our code) generated on the fly using the 'np.zeros' function. 
If you need the non-fire masks you cand use this script to generate masks to the patches in the INPUT_PATCHES_DIR.

Atention: this script DO NOT check if the patch has fire. It generates one maks filled with zeros to each patch in the
INPUT_PATCHES_DIR that do not have a mask in the OUTPUT_DIR. It do not overwrite the masks in the OUTPUT_DIR.

"""

import sys
import os
import rasterio
import numpy as np
from glob import glob
from tqdm import tqdm

IMAGE_SIZE = (256, 256)

INPUT_PATCHES_DIR = '../../dataset/manual_annotations/patches/landsat_patches' 
OUTPUT_DIR = '../../dataset/manual_annotations/patches/manual_annotations_patches'

MASK_NOTATION = 'v1'
NON_FIRE_MASK = np.zeros(IMAGE_SIZE, dtype=np.uint8)

if __name__ == '__main__':


    patches_paths = glob(os.path.join(INPUT_PATCHES_DIR, '*.tif'))
    print(f'Num. Patches: {len(patches_paths)}')

    for patch_path in tqdm(patches_paths):
        patch_name = os.path.basename(patch_path)
        mask_path = os.path.join(OUTPUT_DIR, patch_name.replace('_RT_', f'_RT_{MASK_NOTATION}_'))

        if os.path.exists(mask_path):
            continue

        with rasterio.open(patch_path, 'r') as src:
            meta = src.meta
               
        meta.update(count=1)
        meta.update(dtype=rasterio.uint8)
        with rasterio.open(mask_path, 'w+', **meta) as dst:
            dst.write_band(1, NON_FIRE_MASK)


    print('Done!')