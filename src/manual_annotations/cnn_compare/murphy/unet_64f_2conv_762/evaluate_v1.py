import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from scipy.ndimage.measurements import label
import random
import glob
import time
import pandas as pd

start = time.time()

# Schroeder, Murphy or Kumar-Roy
MASK_ALGORITHM = 'Murphy'
IMAGE_SIZE = (256, 256)

RANDOM_STATE = 42

OUTPUT_DIR = './log'

TH_FIRE = 0.25

def statistics (y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp

def statistics2 (y_true, y_pred):
    py_actu = pd.Series(y_true, name='Actual')
    py_pred = pd.Series(y_pred, name='Predicted')
    df_confusion = pd.crosstab(py_actu, py_pred)
    return df_confusion[0][0], df_confusion[0][1], df_confusion[1][0], df_confusion[1][1]

def statistics3 (y_true, y_pred):
    y_pred_neg = 1 - y_pred
    y_expected_neg = 1 - y_true

    tp = np.sum(y_pred * y_true)
    tn = np.sum(y_pred_neg * y_expected_neg)
    fp = np.sum(y_pred * y_expected_neg)
    fn = np.sum(y_pred_neg * y_true)
    return tn, fp, fn, tp

def jaccard3 (im1, im2):
    """
    Computes the Jaccard metric, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    jaccard : float
        Jaccard metric returned is a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
    
    Notes
    -----
    The order of inputs for `jaccard` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)

    union = np.logical_or(im1, im2)

    jaccard_fire = intersection.sum() / float(union.sum())

    im1 = np.logical_not(im1)
    im2 = np.logical_not(im2)

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)

    union = np.logical_or(im1, im2)

    jaccard_non_fire = intersection.sum() / float(union.sum())

    jaccard_avg = (jaccard_fire + jaccard_non_fire)/2

    return jaccard_fire, jaccard_non_fire, jaccard_avg

def pixel_accuracy (y_true, y_pred):
    sum_n = np.sum(np.logical_and(y_pred, y_true))
    sum_t = np.sum(y_true)
 
    if (sum_t == 0):
        pixel_accuracy = 0
    else:
        pixel_accuracy = sum_n / sum_t
    return pixel_accuracy    

def connected_components (array):
   structure = np.ones((3, 3), dtype=np.int)  # 8-neighboorhood
   labeled, ncomponents = label(array, structure)
   return labeled

def region_relabel (y_true, y_pred):
   index = 0
   for p in y_true:
      if y_pred[index] == 1:
          if y_true[index] > 0:
              y_pred[index] = y_true[index]
          else:
              y_pred[index] = 999
      index += 1


y_pred_all_v1 = []
y_true_all_v1 = []
y_pred_all_multi_v1 = []
y_true_all_multi_v1 = []

jaccard_score_sum_v1 = 0
f1_score_sum_v1 = 0
pixel_accuracy_sum_v1 = 0

jaccard_score_sum_multi_v1 = 0
pixel_accuracy_sum_multi_v1 = 0

nsum_v1 = 0

num_predicted_samples = 0
step = 0
num_images_fire = 0
        
txts_mask_path = sorted(glob.glob(os.path.join(OUTPUT_DIR, MASK_ALGORITHM, 'arrays', 'grd_*.txt')))
txts_pred_path = sorted(glob.glob(os.path.join(OUTPUT_DIR, MASK_ALGORITHM, 'arrays', 'det_*.txt')))

print('# Masks: {}'.format(len(txts_mask_path)))
print('# Pred.: {}'.format(len(txts_pred_path)))

steps = len(txts_mask_path)
    
for txt_mask_path, txt_pred_path in zip(txts_mask_path, txts_pred_path):
    
    try:

        if txt_mask_path.replace('grd', 'det').replace('_{}'.format(MASK_ALGORITHM), '') != txt_pred_path:
            print('[ERROR] Dont match {} - {}'.format(txt_mask_path, txt_pred_path))
            sys.exit()

        y_true = np.loadtxt(txt_mask_path, usecols=range(IMAGE_SIZE[1]))
        y_pred = np.loadtxt(txt_pred_path, usecols=range(IMAGE_SIZE[1]))

        y_true = np.array(y_true, dtype=np.uint8)
        y_pred = np.array(y_pred, dtype=np.uint8)

        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        y_pred_all_v1.append(y_pred)
        y_true_all_v1.append(y_true)

        jaccard_score_v1 = jaccard_score(y_true, y_pred, average='macro')
        jaccard_score_sum_v1 = jaccard_score_sum_v1 + jaccard_score_v1

        f1_score_v1 = f1_score(y_true, y_pred)
        f1_score_sum_v1 = f1_score_sum_v1 + f1_score_v1

        pixel_accuracy_v1 = pixel_accuracy(y_true, y_pred)
        pixel_accuracy_sum_v1 = pixel_accuracy_sum_v1 + pixel_accuracy_v1

        nsum_v1 = nsum_v1 + 1

        count_fire_pixel_mask = np.sum(y_true)
        count_fire_pixel_pred = np.sum(y_pred)
        
        step += 1
        if step%100 == 0:
            print('Step {} of {}'.format(step, steps)) 
            
    except Exception as e:
        print(e)
        
        with open(os.path.join(OUTPUT_DIR, "error_log.txt"), "a+") as myfile:
            myfile.write(str(e))
    


y_pred_all_v1 = np.array(y_pred_all_v1, dtype=np.uint8)
y_pred_all_v1 = y_pred_all_v1.flatten()

y_true_all_v1 = np.array(y_true_all_v1, dtype=np.uint8)
y_true_all_v1 = y_true_all_v1.flatten()

tn, fp, fn, tp = statistics3 (y_true_all_v1, y_pred_all_v1)

P = float(tp)/(tp + fp)
R = float(tp)/(tp + fn)
IoU = float(tp)/(tp+fp+fn)
F = (2 * P * R)/(P + R)
print ('P: :', P, ' R: ', R, ' IoU: ', IoU, ' F-score: ', F)
