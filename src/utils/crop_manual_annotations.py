import sys
sys.path.append('../')
import os
from glob import glob
from landsat.create_masks import get_split

INPUT_ANNOTATION_SCENE_DIR = '../../dataset/manual_annotations/scenes'
OUTPUT_ANNOTATION_PATCHES = '../../dataset/manual_annotations/patches'

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def split_images(input_dir, output_dir):
    print('Croping images in {} to {}'.format(input_dir, output_dir))
    create_folder(output_dir)

    if not output_dir.endswith('/'):
        output_dir += '/'

    images = glob(os.path.join(input_dir, '*.TIF')) + glob(os.path.join(input_dir, '*.tif'))
    print('Num. images: {}'.format(len(images)))

    for image in images:
        _ = get_split(image, output_dir)


if __name__ == '__main__':
    images_dir = os.path.join(INPUT_ANNOTATION_SCENE_DIR, 'images')
    masks_dir = os.path.join(INPUT_ANNOTATION_SCENE_DIR, 'masks')
    manual_annotation_dir = os.path.join(INPUT_ANNOTATION_SCENE_DIR, 'annotations')

    output_images_dir = os.path.join(OUTPUT_ANNOTATION_PATCHES, 'images')
    output_masks_dir = os.path.join(OUTPUT_ANNOTATION_PATCHES, 'masks')
    output_manual_annotation_dir = os.path.join(OUTPUT_ANNOTATION_PATCHES, 'annotations')

    split_images(images_dir, output_images_dir)
    split_images(masks_dir, output_masks_dir)
    split_images(manual_annotation_dir, output_manual_annotation_dir)




