import sys
import os
import pandas as pd
from shutil import copy2
from tqdm import tqdm


IMAGES_FOLDER = '/home/andre/tif_patches/patches/'
MASKS_FOLDER = '/home/andre/mask_patches/patches/'

OUTPUT_FOLDER = '/../../journal-dataset-and-code/dataset'
OUTPUT_IMAGES = os.path.join(OUTPUT_FOLDER, 'images', 'patches')
OUTPUT_MASKS = os.path.join(OUTPUT_FOLDER, 'masks', 'patches')

CSV_IMAGES = [
    './dataset/goli/images_test.csv',
    './dataset/goli/images_train.csv',
    './dataset/goli/images_val.csv',

    './dataset/murphy/images_val.csv',
    './dataset/murphy/images_test.csv',
    './dataset/murphy/images_train.csv',

    './dataset/schroeder/images_val.csv',
    './dataset/schroeder/images_test.csv',
    './dataset/schroeder/images_train.csv',
]

CSV_MASKS = [
    './dataset/goli/masks_test.csv',
    './dataset/goli/masks_train.csv',
    './dataset/goli/masks_val.csv',
    
    './dataset/murphy/masks_val.csv',
    './dataset/murphy/masks_test.csv',
    './dataset/murphy/masks_train.csv',

    './dataset/schroeder/masks_val.csv',
    './dataset/schroeder/masks_test.csv',
    './dataset/schroeder/masks_train.csv',
]

def load_csv(csv_list):

    dataframes = []
    for csv_file in csv_list:
        df = pd.read_csv(csv_file)
    
        print('{} - {}'.format(csv_file, len(df.index) ))
        
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True, sort=False)
    return df


if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_IMAGES)
    os.makedirs(OUTPUT_MASKS)


df = load_csv(CSV_IMAGES)
print('Moving images...')
images = df.images.unique()
for image in tqdm(images):
    copy2(os.path.join(IMAGES_FOLDER, image), os.path.join(OUTPUT_IMAGES))

print('Images moved')

df = load_csv(CSV_MASKS)
print('Moving masks...')
masks = df.masks.unique()
for mask in tqdm(masks):
    copy2(os.path.join(MASKS_FOLDER, mask), os.path.join(OUTPUT_MASKS))

print('Masks moved')


print('Total images: {}'.format( len(os.listdir(OUTPUT_IMAGES)) ))
print('Total masks: {}'.format( len(os.listdir(OUTPUT_MASKS)) ))


