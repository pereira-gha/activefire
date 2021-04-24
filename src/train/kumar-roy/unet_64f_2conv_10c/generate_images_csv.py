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

# Schoeder, Murphy or Kumar-Roy
MASK_ALGORITHM = 'Kumar-Roy'

IMAGES_PATH = '/home/andre/tif_patches/patches/'
MASKS_PATH = '/home/andre/mask_patches/patches/'
IMAGES_DATAFRAME = './dataset/images_masks.csv'

RANDOM_STATE = 42

masks = glob.glob(os.path.join(MASKS_PATH, '*{}*.tif'.format(MASK_ALGORITHM)))

print(os.path.join(MASKS_PATH, '*{}*.tif'.format(MASK_ALGORITHM)))
print(len(masks))

with open(IMAGES_DATAFRAME, 'w') as f:
    writer = csv.writer(f, delimiter=',')

    for mask in tqdm(masks):
        _, mask_name = os.path.split(mask)

        image_name = mask_name.replace('_{}_'.format(MASK_ALGORITHM), '_')
        writer.writerow([image_name, mask_name])


train_ratio = 0.4
validation_ratio = 0.1
test_ratio = 0.5


df = pd.read_csv(IMAGES_DATAFRAME, header=None, names=['images', 'masks'])
images_df = df[ ['images'] ]
masks_df = df[ ['masks'] ]

x_train, x_test, y_train, y_test = train_test_split(images_df, masks_df, test_size=1 - train_ratio, random_state=RANDOM_STATE)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=RANDOM_STATE) 

print(len(df.index), len(x_train.index), len(x_val.index), len(x_test.index))

x_train.to_csv('./dataset/images_train.csv', index=False)
y_train.to_csv('./dataset/masks_train.csv', index=False)
x_val.to_csv('./dataset/images_val.csv', index=False)
y_val.to_csv('./dataset/masks_val.csv', index=False)
x_test.to_csv('./dataset/images_test.csv', index=False)
y_test.to_csv('./dataset/masks_test.csv', index=False)
