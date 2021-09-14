import os
import warnings
warnings.filterwarnings("ignore")

import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras import backend as K

# from generator import *
# from models import *
# from metrics import *
# from plot_history import plot_history
import sys
import csv
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

# Schroeder, Murphy or Kumar-Roy
MASK_ALGORITHM = 'Kumar-Roy'

IMAGES_PATH = '../../dataset/images/patches/'
MASKS_PATH = '../../dataset/masks/patches/'
OUTPUT_FOLDER = '../../dataset/'
IMAGES_DATAFRAME = os.path.join(OUTPUT_FOLDER, 'images_masks.csv')

RANDOM_STATE = 42

TRAIN_RATIO = 0.4
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.5

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

masks = glob.glob(os.path.join(MASKS_PATH, '*{}*.tif'.format(MASK_ALGORITHM)))

print(os.path.join(MASKS_PATH, '*{}*.tif'.format(MASK_ALGORITHM)))
print(len(masks))

with open(IMAGES_DATAFRAME, 'w') as f:
    writer = csv.writer(f, delimiter=',')

    for mask in tqdm(masks):
        _, mask_name = os.path.split(mask)

        image_name = mask_name.replace('_{}_'.format(MASK_ALGORITHM), '_')
        writer.writerow([image_name, mask_name])


df = pd.read_csv(IMAGES_DATAFRAME, header=None, names=['images', 'masks'])
images_df = df[ ['images'] ]
masks_df = df[ ['masks'] ]

x_train, x_test, y_train, y_test = train_test_split(images_df, masks_df, test_size=1 - TRAIN_RATIO, random_state=RANDOM_STATE)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=TEST_RATIO/(TEST_RATIO + VALIDATION_RATIO), random_state=RANDOM_STATE) 

print(len(df.index), len(x_train.index), len(x_val.index), len(x_test.index))

x_train.to_csv(os.path.join(OUTPUT_FOLDER, 'images_train.csv'), index=False)
y_train.to_csv(os.path.join(OUTPUT_FOLDER, 'masks_train.csv'), index=False)
x_val.to_csv(os.path.join(OUTPUT_FOLDER, 'images_val.csv'), index=False)
y_val.to_csv(os.path.join(OUTPUT_FOLDER, 'masks_val.csv'), index=False)
x_test.to_csv(os.path.join(OUTPUT_FOLDER, 'images_test.csv'), index=False)
y_test.to_csv(os.path.join(OUTPUT_FOLDER, 'masks_test.csv'), index=False)

print('Done!')