import numpy as np
from scipy.ndimage.measurements import label

def statistics3 (y_true, y_pred):
    y_pred_neg = 1 - y_pred
    y_expected_neg = 1 - y_true

    tp = np.sum(y_pred * y_true)
    tn = np.sum(y_pred_neg * y_expected_neg)
    fp = np.sum(y_pred * y_expected_neg)
    fn = np.sum(y_pred_neg * y_true)
    return tn, fp, fn, tp

def jaccard3 (im1, im2):
   
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
