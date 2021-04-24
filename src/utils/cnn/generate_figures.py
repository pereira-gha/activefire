import numpy as np
import rasterio
import pandas as pd
import os
import sys
import cv2
from glob import glob
import tensorflow as tf
from models import *
from tqdm import tqdm

CUDA_DEVICE = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
    np.random.bit_generator = np.random._bit_generator
except:
    pass

# Numero minimo de pixels para considerar a imagem
MIN_FIRE_PIXELS = 100
TH_FIRE = 0.25

MODELS_CNN = {
    'unet_10b' : {
        'model_name': 'unet',
        'n_channels': 10,
        'n_filters': 64,
    },
    'unet_3b' : {
        'model_name': 'unet',
        'n_channels': 3,
        'n_filters': 64,
    },
    'unet_light' : {
        'model_name': 'unet',
        'n_channels': 3,
        'n_filters': 16,
    },
}

# Parametros para geracao das mascaras
# O dataframe de imagens pode ser apontado para qualquer diretorio do algoritmo - sao todos iguais
PARAMETERS_MASKS = {
    'Schroeder' : {
        'weights': {
            'unet_10b': '../../train/schroeder/unet_64f_2conf_10c/train_output/model_unet_Schroeder_final_weights.h5',
            'unet_3b': '../../train/schroeder/unet_64f_2conf_762/train_output/model_unet_Schroeder_final_weights.h5',
            'unet_light': '../../train/schroeder/unet_16f_2conf_762/train_output/model_unet_Schroeder_final_weights.h5',
        }
    },
    'Murphy' : {
        'weights': {
            'unet_10b': '../../train/murphy/unet_64f_2conf_10c/train_output/model_unet_Murphy_final_weights.h5',
            'unet_3b': '../../train/murphy/unet_64f_2conf_762/train_output/model_unet_Murphy_final_weights.h5',
            'unet_light': '../../train/murphy/unet_16f_2conf_762/train_output/model_unet_Murphy_final_weights.h5',
        }
    },
    'Kumar-Roy' : {
        'weights': {
            'unet_10b': '../../train/kumar-roy/unet_64f_2conf_10c/train_output/model_unet_Kumar-Roy_final_weights.h5',
            'unet_3b': '../../train/kumar-roy/unet_64f_2conf_762/train_output/model_unet_Kumar-Roy_final_weights.h5',
            'unet_light': '../../train/kumar-roy/unet_16f_2conf_762/train_output/model_unet_Kumar-Roy_final_weights.h5',
        }
    },
    'Intersection' : {
        'weights': {
            'unet_10b': '../../train/intersection/unet_64f_2conf_10c/train_output/model_unet_Intersection_final_weights.h5',
            'unet_3b': '../../train/intersection/unet_64f_2conf_762/train_output/unet_64f_2conv_762/weights/model_unet_Intersection_final_weights.h5',
            'unet_light': '../../train/intersection/unet_16f_2conf_762/train_output/unet_16f_2conv_762/weights/model_unet_Intersection_final_weights.h5',
        }
    },
    'Voting' : {
        'weights': {
            'unet_10b': '../../train/voting/unet_64f_2conf_10c/train_output/model_unet_Voting_final_weights.h5',
            'unet_3b': '../../train/voting/unet_64f_2conf_762/train_output/model_unet_Voting_final_weights.h5',
            'unet_light': '../../train/voting/unet_16f_2conf_762/train_output/model_unet_Voting_final_weights.h5',
        }
    },
}

IMAGE_SIZE = (256, 256)
MAX_PIXEL_VALUE = 65535

OUTPUT_PATH = os.path.join('./figuras/RGB/', str(MIN_FIRE_PIXELS))
IMAGES_PATH = '../../../dataset/images/patches/'
MANUAL_ANNOTATION_PATH = '../../../dataset/groundtruth/patches/'
ALGORITHM_MASKS_PATH = '../../../dataset/masks/patches/'

IMAGE_NAME = 'LC08_L1TP_193029_20200914_20200914_01_RT'


# Desconsideras as mascaras que contem os seguintes itens no nome
IGNORE_MANUAL_ANNOTATION_WITH_STR = ['v2']
REMOVE_STR_FROM_MASK_NAME = [ 'v1' ]


def load_manual_annotation_masks_as_dataframe():
    df = load_path_as_dataframe(MANUAL_ANNOTATION_PATH)
      # remove mascaras com as strings definidas
    for ignore_mask_with_str in IGNORE_MANUAL_ANNOTATION_WITH_STR:
        df = df[~df.original_name.str.contains(ignore_mask_with_str)]

    return df

def load_path_as_dataframe(mask_path):
    masks = glob(os.path.join(mask_path, '{}*.tif'.format(IMAGE_NAME)))

    print('Carregando diretorio: {}'.format(mask_path))
    print('Total de máscaras no diretórios: {}'.format(len(masks)))

    df = pd.DataFrame(masks, columns=['image_path'])
    # recupera o nome da máscara pelo caminho dela
    df['original_name'] = df.image_path.apply(os.path.basename)
    # remove o algoritmo gerador da mascara do nome dela
    df['image_name'] = df.original_name.apply(remove_algorithms_name)

    return df

def remove_algorithms_name(mask_name):
    """Remove o nome dos algoritmos do nome da máscara"""

    #algorithms_name = MASKS_ALGORITHMS + ['mask']
    #algorithms_name = MASKS_ALGORITHMS + ['762_mask']
    # algorithms_name = MASKS_ALGORITHMS + ['v2']
    
    for algorithm in REMOVE_STR_FROM_MASK_NAME:
        mask_name = mask_name.replace('_{}'.format(algorithm), '')

    return mask_name

def merge_dataframes(df_manual, df_algorithm):
    return pd.merge(df_manual, df_algorithm, on = 'image_name', how='outer')


def get_img_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img

def get_img_762bands(path):
    img = rasterio.open(path).read((7,6,2)).transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img
    
def get_mask_arr(path):
    """ Abre a mascara como array"""
    with rasterio.open(path) as src:
        img = src.read().transpose((1, 2, 0))
        seg = np.array(img, dtype=int)

        return seg[:, :, 0]

if __name__ == '__main__':

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    df = load_path_as_dataframe(IMAGES_PATH)
    i = 0
    
    paths_with_masks = []
    samples = {}
    for algorithm in PARAMETERS_MASKS:
        parameters = PARAMETERS_MASKS[algorithm]

        # df_manual = load_manual_annotation_masks_as_dataframe()
        # df = merge_dataframes(df_manual, df_images)

   

        # Para cara modelo de rede carrega os pesos para um algoritmo e roda a inferencia para as imagens
        for model_cnn in MODELS_CNN:
            parameters_cnn = MODELS_CNN[model_cnn]
            
            model = get_model('unet', input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=parameters_cnn['n_filters'], n_channels=parameters_cnn['n_channels'])
            
            print('Loading weghts...')
            print(parameters['weights'][model_cnn])
            model.load_weights( parameters['weights'][model_cnn] )
            
            # Define a forma de abrir a imagem (10 bandas ou 3 bands)
            open_image = get_img_arr
            if parameters_cnn['n_channels'] == 3:
                open_image = get_img_762bands

            for index, row in tqdm(df.iterrows()):
                
                image_path = row['image_path']
                image_name = row['image_name']
                

                # Verifica a pasta que a imagem deve ser salva
                if row['original_name'] in samples:
                    output_dir = samples[row['original_name']]
                else:
                    output_dir = os.path.join(OUTPUT_PATH, 'sample_{}'.format(i))
                    samples[row['original_name']] = output_dir
                    i += 1


                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Faz a inferencia
                img = open_image(image_path) 
                y_pred = model.predict(np.array( [ img ] ), batch_size=1)

                y_pred = y_pred[0, :, :, 0] > TH_FIRE
                y_pred = np.array(y_pred * 255, dtype=np.uint8)
                

                # Salva a predicao
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                output_name = '{}_{}'.format(image_name, model_cnn)
                output_prediction = os.path.join(output_dir, '{}_{}.png'.format(algorithm, output_name))
                cv2.imwrite(output_prediction, y_pred)
                
                # # Se nao existir imagem 762 na pasta gera uma nova imagem
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                output_image = os.path.join(output_dir, '{}_762.png'.format(image_name))
                if not os.path.exists(output_image):    
                    image_762 = get_img_762bands(image_path)
                    image_762 = np.array(image_762 * 255, dtype=np.uint8)
   
                    cv2.imwrite(output_image, cv2.cvtColor(image_762, cv2.COLOR_RGB2BGR))

                # Salva a mascara do algoritmo na pasta
                paths_with_masks.append(output_dir)

                mask_alg_path = os.path.join(ALGORITHM_MASKS_PATH, row['image_name'].replace('_RT_', '_RT_{}_'.format(algorithm)) )
                if not os.path.exists(mask_alg_path):
                    continue
                
                mask_name = os.path.splitext(os.path.basename(mask_alg_path))[0]
                mask_output = os.path.join(output_dir, '{}.png'.format(mask_name))
                if not os.path.exists(mask_output):
                    mask_alg = get_mask_arr(mask_alg_path)
                    mask_alg = np.array(mask_alg * 255, dtype=np.uint8)
                    cv2.imwrite(mask_output, mask_alg)


        paths_with_masks = list(set(paths_with_masks))
        print('{} - Paths with Masks: {}'.format(algorithm, len(paths_with_masks)))
    print(paths_with_masks)




