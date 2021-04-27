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

MANUAL_ANNOTATION_PATH = '../../dataset/manual_annotations/patches/manual_annotations_patches'
MASKS_PATH = '../../dataset/manual_annotations/patches/masks_patches'

# Para facilitar, se for teste com um diretório que não segue o padrão dos nomes das máscaras: *(Kumar-Roy, Murphy, Schroeder)*.tif
# Marque a flag como True. Com isso o diretório com as imagens será retornado em um dataframe, se False serão os dataframes serão separados com base no nome 
IS_TEST = False

MASKS_ALGORITHMS = ['Schroeder', 'Murphy', 'Kumar-Roy', 'Intersection', 'Voting']

# Desconsideras as mascaras que contem os seguintes itens no nome
IGNORE_MANUAL_ANNOTATION_WITH_STR = ['v2']

# Remove as strings definidas do nome das mascas para realizar um match entre mascara e imagem analisada
REMOVE_STR_FROM_MASK_NAME = MASKS_ALGORITHMS + ['v1']

OUTPUT_DIR = './log'
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

        if not os.path.exists(os.path.join(OUTPUT_DIR, algorithm, 'arrays')):
            os.makedirs(os.path.join(OUTPUT_DIR, algorithm, 'arrays'))

        df_algorithm = dataframes[ algorithm ]
        
        print('Algorithm: {} - Num. Masks: {}'.format(algorithm, len(df_algorithm.index)))

        df = merge_dataframes(df_manual, df_algorithm)

        for index, row in tqdm(df.iterrows()):

            y_true = get_mask(row['masks_path_x'])
            y_pred = get_mask(row['masks_path_y'])

            image_path = row['image_name']

            mask_name = os.path.splitext(os.path.basename(image_path))[0]
            image_name = os.path.splitext(os.path.basename(image_path))[0]

            txt_mask_path = os.path.join(OUTPUT_DIR, algorithm, 'arrays', 'grd_' + mask_name + '.txt') 
            txt_pred_path = os.path.join(OUTPUT_DIR, algorithm, 'arrays', 'det_' + image_name + '.txt') 

            np.savetxt(txt_mask_path, y_true.astype(int), fmt='%i')
            np.savetxt(txt_pred_path, y_pred.astype(int), fmt='%i')
