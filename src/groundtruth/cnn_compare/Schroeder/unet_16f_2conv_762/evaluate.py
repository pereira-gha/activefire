import pandas as pd
import numpy as np
import rasterio
import os
from glob import glob
import sys
from models import *
from metrics import *
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from tqdm import tqdm

MANUAL_ANNOTATION_PATH = '../../../../../dataset/groundtruth/patches/'
IMAGES_PATH = '../../../../../dataset/groundtruth/images_patches'

CUDA_DEVICE = 0

# 10 or 3
N_CHANNELS = 3
# 16 or 64
N_FILTERS = 16

MASK_ALGORITHM = 'Schroeder'
MODEL_NAME = 'unet'
WEIGHTS_FILE = './weights/model_{}_{}_final_weights.h5'.format(MODEL_NAME, MASK_ALGORITHM)

MASKS_ALGORITHMS = ['Schroeder', 'Murphy', 'GOLI_v2']

# Desconsideras as mascaras que contem os seguintes itens no nome
IGNORE_MASKS_WITH_STR = ['v2']

# Remove as strings definidas do nome das mascas para realizar um match entre mascara e imagem analisada
REMOVE_STR_FROM_MASK_NAME = MASKS_ALGORITHMS + ['v1' , 'noFire']

TH_FIRE = 0.25
IMAGE_SIZE = (256, 256)
MAX_PIXEL_VALUE = 65535 # Max. pixel value, used to normalize the image


def load_path_as_dataframe(mask_path):
    masks = glob(os.path.join(mask_path, '*.tif'))

    print('Carregando diretório: {}'.format(mask_path))
    print('Total de máscaras no diretórios: {}'.format(len(masks)))

    df = pd.DataFrame(masks, columns=['masks_path'])
    # recupera o nome da máscara pelo caminho dela
    df['original_name'] = df.masks_path.apply(os.path.basename)
    # remove o algoritmo gerador da mascara do nome dela
    df['image_name'] = df.original_name.apply(remove_algorithms_name)

    # remove mascaras com as strings definidas
    for ignore_mask_with_str in IGNORE_MASKS_WITH_STR:
        df = df[~df.original_name.str.contains(ignore_mask_with_str)]

    return df

def remove_algorithms_name(mask_name):
    """Remove o nome dos algoritmos do nome da máscara"""

    #algorithms_name = MASKS_ALGORITHMS + ['mask' , 'noFire']
    # algorithms_name = MASKS_ALGORITHMS + ['v2' , 'noFire']

    for algorithm in REMOVE_STR_FROM_MASK_NAME:
        mask_name = mask_name.replace('_{}'.format(algorithm), '')

    return mask_name

def merge_dataframes(df_manual, df_images):
    return pd.merge(df_manual, df_images, on = 'image_name', how='outer')



def open_manual_annotation(path):
    if type(path) != str:
        mask = np.zeros(IMAGE_SIZE)
    else:
        mask = get_mask_arr(path)

    return np.array(mask, dtype=np.uint8)

def get_mask_arr(path):
    """ Abre a mascara como array"""
    with rasterio.open(path) as src:
        img = src.read().transpose((1, 2, 0))
        seg = np.array(img, dtype=int)

        return seg[:, :, 0]


def get_img_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img


def get_img_762bands(path):
    img = rasterio.open(path).read((7,6,2)).transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

    try:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        K.set_session(sess)
        np.random.bit_generator = np.random._bit_generator
    except:
        pass

    # Define a função de abertura de imagens
    open_image = get_img_arr
    if N_CHANNELS == 3:
        open_image = get_img_762bands



    model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)

    model.summary()
    print('Loading weghts...')
    model.load_weights(WEIGHTS_FILE)
    print('Weights Loaded')


    df_manual = load_path_as_dataframe(MANUAL_ANNOTATION_PATH)
    df_images = load_path_as_dataframe(IMAGES_PATH)

    df = merge_dataframes(df_manual, df_images)
    print('Total de Imagens: {}'.format(len(df.index)))

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

    for index, row in tqdm(df.iterrows()):
        # print(row['masks_path_y'])
        y_true = open_manual_annotation(row['masks_path_x'])
        img = open_image(row['masks_path_y'])


        y_pred = model.predict(np.array( [img] ), batch_size=1)
        y_pred = y_pred[0, :, :, 0] > TH_FIRE
        y_pred = np.array(y_pred, dtype=np.uint8)
        

        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        y_pred_all_v1.append(y_pred)
        y_true_all_v1.append(y_true)
        

        jaccard_score_v1 = jaccard_score(y_true, y_pred, average='macro')
        jaccard_score_sum_v1 = jaccard_score_sum_v1 + jaccard_score_v1

        f1_score_v1 = f1_score(y_true, y_pred, zero_division=0)
        f1_score_sum_v1 = f1_score_sum_v1 + f1_score_v1

        pixel_accuracy_v1 = pixel_accuracy(y_true, y_pred)
        pixel_accuracy_sum_v1 = pixel_accuracy_sum_v1 + pixel_accuracy_v1

        nsum_v1 = nsum_v1 + 1


    y_pred_all_v1 = np.array(y_pred_all_v1, dtype=np.uint8)
    y_pred_all_v1 = y_pred_all_v1.flatten()
    y_true_all_v1 = np.array(y_true_all_v1, dtype=np.uint8)
    y_true_all_v1 = y_true_all_v1.flatten()

    print('y_true_all_v1 shape: ', y_true_all_v1.shape)
    print('y_pred_all_v1 shape: ', y_pred_all_v1.shape)

    print ('-------------------- ALL IMAGE-BY-IMAGE (BINARY) - V1')
    print ('mIoU (Jaccard-Average/Fire & Non-Fire):', float(jaccard_score_sum_v1)/nsum_v1)
    print ('F1-score (Dice/Fire & Non-Fire):', float(f1_score_sum_v1)/nsum_v1)
    print ('Pixel-accuracy (IoU fire):', float(pixel_accuracy_sum_v1)/nsum_v1)

    print ('-------------------- ALL (BINARY) - V1')
    print ('mIoU (Jaccard3):', jaccard3 (y_true_all_v1, y_pred_all_v1))

    tn, fp, fn, tp = statistics3 (y_true_all_v1, y_pred_all_v1)
    print ('Statistics3 (tn, fp, fn, tp):', tn, fp, fn, tp)

    P = float(tp)/(tp + fp)
    R = float(tp)/(tp + fn)
    IoU = float(tp)/(tp+fp+fn)
    Acc = float((tp+tn))/(tp+tn+fp+fn)
    F = (2 * P * R)/(P + R)
    print ('P: :', P, ' R: ', R, ' IoU: ', IoU, ' Acc: ', Acc, ' F-score: ', F)




