import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from metrics import *
import os
import sys
import rasterio

MANUAL_ANNOTATION_PATH = '../../dataset/groundtruth/patches/'
MASKS_PATH = '../../dataset/groudtruth/masks/'

# Para facilitar, se for teste com um diretório que não segue o padrão dos nomes das máscaras: *(GOLI_v2, Murphy, Schroeder)*.tif
# Marque a flag como True. Com isso o diretório com as imagens será retornado em um dataframe, se False serão os dataframes serão separados com base no nome 
IS_TEST = False

MASKS_ALGORITHMS = ['Schroeder', 'Murphy', 'GOLI_v2']

# Desconsideras as mascaras que contem os seguintes itens no nome
IGNORE_MANUAL_ANNOTATION_WITH_STR = ['v2']

# Remove as strings definidas do nome das mascas para realizar um match entre mascara e imagem analisada
REMOVE_STR_FROM_MASK_NAME = MASKS_ALGORITHMS + [ 'v1' ]


IMAGE_SIZE = (256, 256)

def load_manual_annotation_masks_as_dataframe():
    df = load_path_as_dataframe(MANUAL_ANNOTATION_PATH)
      # remove mascaras com as strings definidas
    for ignore_mask_with_str in IGNORE_MANUAL_ANNOTATION_WITH_STR:
        df = df[~df.original_name.str.contains(ignore_mask_with_str)]

    return df

def load_masks_path_as_dataframes():

    df = load_path_as_dataframe(MASKS_PATH)

    #! Para testar retorna o diretório de imagens sem separar pelo nome de máscara
    if IS_TEST:
        return {'Teste': df}

    print('Spliting masks...')
    total = 0
    dataframes = {}

    # separa as imagens com base no nome dos algoritmos geradores máscaras
    for i, algorithm in enumerate(MASKS_ALGORITHMS):
    
        dataframes[algorithm] = df[ df['original_name'].str.contains(algorithm) ]

        num_images = len(dataframes[algorithm].index)
        total += num_images
        print('{} - images: {}'.format(algorithm, num_images))

    return dataframes


def load_path_as_dataframe(mask_path):
    masks = glob(os.path.join(mask_path, '*.tif'))

    print('Loading path: {}'.format(mask_path))
    print('Files found: {}'.format(len(masks)))

    df = pd.DataFrame(masks, columns=['masks_path'])
    # recupera o nome da máscara pelo caminho dela
    df['original_name'] = df.masks_path.apply(os.path.basename)
    # remove o algoritmo gerador da mascara do nome dela
    df['image_name'] = df.original_name.apply(remove_algorithms_name)

    return df

def remove_algorithms_name(mask_name):
    """Remove o nome dos algoritmos do nome da máscara"""
    
    for algorithm in REMOVE_STR_FROM_MASK_NAME:
        mask_name = mask_name.replace('_{}'.format(algorithm), '')

    return mask_name

def merge_dataframes(df_manual, df_algorithm):
    return pd.merge(df_manual, df_algorithm, on = 'image_name', how='outer')


def get_mask(path):
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


if __name__ == '__main__':

    df_manual = load_manual_annotation_masks_as_dataframe()
    dataframes = load_masks_path_as_dataframes()

    print('Num. Groundtruth: {}'.format( len( df_manual.index )) )

    for algorithm in dataframes:

        df_algorithm = dataframes[ algorithm ]
        
        print('Algorithm: {} - Num. Masks: {}'.format(algorithm, len(df_algorithm.index)))

        df = merge_dataframes(df_manual, df_algorithm)

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

        for index, row in tqdm(df.iterrows()):

            y_true = get_mask(row['masks_path_x'])
            y_pred = get_mask(row['masks_path_y'])

            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
            
            ttn, tfp, tfn, ttp = statistics3 (y_true, y_pred)
            print (row['masks_path_x'], row['masks_path_y'], 'tn: ', ttn, 'fp: ', tfp, 'fn: ', tfn, 'tp: ', ttp)

            y_pred_all_v1.append(y_pred)
            y_true_all_v1.append(y_true)
            

            jaccard_score_v1 = jaccard_score(y_true, y_pred, average='macro')
            jaccard_score_sum_v1 = jaccard_score_sum_v1 + jaccard_score_v1

            f1_score_v1 = f1_score(y_true, y_pred)
            f1_score_sum_v1 = f1_score_sum_v1 + f1_score_v1

            pixel_accuracy_v1 = pixel_accuracy(y_true, y_pred)
            pixel_accuracy_sum_v1 = pixel_accuracy_sum_v1 + pixel_accuracy_v1

            nsum_v1 = nsum_v1 + 1

        y_pred_all_v1 = np.array(y_pred_all_v1, dtype=np.uint8)
        y_pred_all_v1 = y_pred_all_v1.flatten()
        y_true_all_v1 = np.array(y_true_all_v1, dtype=np.uint8)
        y_true_all_v1 = y_true_all_v1.flatten()

        print('Resultados do Algoritmo: {}'.format(algorithm))

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




    


