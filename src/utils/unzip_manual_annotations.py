import zipfile
from glob import glob
import os
import shutil

INPUT_DIR = '../../dataset/manual_annotations/compressed'
OUTPUT_DIR = '../../dataset/manual_annotations/'

# Set to true if you are unziping the complete scene from landsat, not the patches
UNZIP_SCENES = False

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def unzip_to_folder(input_zip, output_folder):
    print('Unziping: {}'.format(input_zip))
    with zipfile.ZipFile(input_zip, 'r') as zip_ref:
        print('Num ziped Files: {}'.format(len(zip_ref.namelist())))

        zip_ref.extractall(output_folder)

if UNZIP_SCENES:

    input_dir = INPUT_DIR
    output_dir = os.path.join(OUTPUT_DIR, 'scenes')
    create_folder(output_dir)

    images_zip_name = 'landsat_scenes.zip'
    manual_annotations_zip_name = 'manual_annotations_scenes.zip'
    masks_zip_name = 'masks_scenes.zip'


    output_images_dir = os.path.join(output_dir, 'images')
    create_folder(output_images_dir)

    output_manual_annotations_dir = os.path.join(output_dir, 'annotations')
    create_folder(output_manual_annotations_dir)

    output_masks_dir = os.path.join(output_dir, 'masks')
    create_folder(output_masks_dir)

else:
    input_dir = INPUT_DIR
    output_dir = os.path.join(OUTPUT_DIR, 'patches')
    create_folder(output_dir)

    images_zip_name = 'landsat_patches.zip'
    manual_annotations_zip_name = 'manual_annotations_patches.zip'
    masks_zip_name = 'masks_patches.zip'

    output_images_dir = output_dir
    output_manual_annotations_dir = output_dir
    output_masks_dir = output_dir

unzip_to_folder(os.path.join(input_dir, images_zip_name), output_images_dir)
unzip_to_folder(os.path.join(input_dir, manual_annotations_zip_name), output_manual_annotations_dir)
unzip_to_folder(os.path.join(input_dir, masks_zip_name), output_masks_dir)
