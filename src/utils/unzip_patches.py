import zipfile
from glob import glob
import os

IMAGES_PATH = '../../dataset/images'
MASKS_PATH = '../../dataset/masks'


images_zips = glob(os.path.join(IMAGES_PATH, '*.zip'))

for image_zip in images_zips:
    with zipfile.ZipFile(image_zip, 'r') as zip_ref:
        zip_ref.extractall(IMAGES_PATH)




masks_zips = glob(os.path.join(MASKS_PATH, '*.zip'))
for mask_zip in masks_zips:
    with zipfile.ZipFile(mask_zip, 'r') as zip_ref:
        zip_ref.extractall(MASKS_PATH)

