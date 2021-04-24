"""
Use this script to unzip the weigts and put it on the spected folders.
"""
import os
import sys
from glob import glob
import zipfile
import shutil

# Change this constant to the folder where you put the ziped weights
ZIPED_WEIGHTS_DIR = '../../weights'

# If you need change the paths to your custom structure
TRAIN_DIR = '../train/'
UNZIP_TO_MANUAL_ANNOTATIONS_CNN = '../manual_annotations/cnn_compare'

# Set to true where you want to put a copy of the unziped weights
UNZIP_TO_TRAIN = True
UNZIP_TO_MANUAL_ANNOTATNIOS_CNN = True


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def unzip_to_folder(input_zip, output_folder):
    print('Unziping: {}'.format(input_zip))
    with zipfile.ZipFile(input_zip, 'r') as zip_ref:
        zip_ref.extractall(output_folder)

ziped_files = glob(os.path.join(ZIPED_WEIGHTS_DIR, '*.zip'))

tmp_dir = os.path.join(ZIPED_WEIGHTS_DIR, 'tmp')
create_folder(tmp_dir)

for ziped_file in ziped_files:
    unzip_to_folder(ziped_file, tmp_dir)

algorithms = os.listdir(tmp_dir)

for algorithm in algorithms:
    
    base_path = os.path.join(tmp_dir, algorithm)
    architectures = os.listdir(base_path)
    for architecture in architectures:
        weight_file_name = 'model_unet_{}_final_weights.h5'
        if algorithm == 'kumar-roy':
            weight_file_name = weight_file_name.format('Kumar-Roy')
        else:
            weight_file_name = weight_file_name.format(algorithm.capitalize())
        
        weight_path = os.path.join(base_path, architecture, weight_file_name)
        
        if UNZIP_TO_TRAIN:
            target_folder = os.path.join(TRAIN_DIR, algorithm, architecture, 'train_output')
            create_folder(target_folder)

            print('Coping {} to {}'.format(weight_path, target_folder))
            shutil.copy2(weight_path, target_folder)

        if UNZIP_TO_MANUAL_ANNOTATNIOS_CNN:
            target_folder = os.path.join(UNZIP_TO_MANUAL_ANNOTATIONS_CNN, algorithm, architecture, 'weights')
            create_folder(target_folder)

            print('Coping {} to {}'.format(weight_path, target_folder))
            shutil.copy2(weight_path, target_folder)

shutil.rmtree(tmp_dir)
