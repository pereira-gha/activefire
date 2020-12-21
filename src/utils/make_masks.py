""" Gera novas máscaras a partir da combinação das máscaras geradas pelos algoritmos da literatura
As novas máscaras são geradas a partir da união, intersecção e votação das máscaras da literatura
Caso uma máscara não exista para um determinado algoritmo é gerado um valor "don't care" como máscara

"""


import pandas as pd
import numpy as np
import rasterio
from glob import glob
from functools import reduce
from tqdm import tqdm
import cv2
import os
import sys

MASKS_DIR = '../../dataset/masks'

MASKS_ALGORITHMS = ['Schroeder', 'Murphy', 'GOLI_v2']

NUM_VOTINGS = 2

IMAGE_SIZE = (256, 256)

def load_images(path):
    masks = glob(os.path.join(path, '*.png'))

    images = []
    for mask in masks:
        for algorithm in MASKS_ALGORITHMS:
            if '_RT_{}'.format(algorithm) in mask:
                images.append(mask)

    return images


def remove_algorithms_name(mask_name):
    """Remove o nome dos algoritmos do nome da máscara"""

    for algorithm in MASKS_ALGORITHMS:
        print(mask_name)
        mask_name = mask_name.replace('_{}'.format(algorithm), '')
    return mask_name


def make_intersection_masks(images):

    # mascara "dont care" toda Verdadeira
    final_mask = (np.ones(IMAGE_SIZE) == 1)
    
    for image in images:
        mask  = get_mask_arr(image)
        # intersecao das máscaras
        final_mask = np.logical_and(final_mask, mask)

    
    output_path = os.path.dirname(os.path.abspath(images[0]))
    image_name = os.path.basename(images[0])
    for alg in MASKS_ALGORITHMS:
        image_name = image_name.replace('_RT_{}'.format(alg), '_RT_Intersection')

    output_path = os.path.join(output_path, image_name)
    write_mask(output_path, final_mask)
    
    print('Máscaras de Instersecção criadas')

def make_union_masks(images):

    final_mask = (np.zeros(IMAGE_SIZE) == 1)
    
    for image in images:
        mask  = get_mask_arr(image)
        # intersecao das máscaras
        final_mask = np.logical_or(final_mask, mask)

    
    output_path = os.path.dirname(os.path.abspath(images[0]))
    image_name = os.path.basename(images[0])
    for alg in MASKS_ALGORITHMS:
        image_name = image_name.replace('_RT_{}'.format(alg), '_RT_Union')

    output_path = os.path.join(output_path, image_name)
    write_mask(output_path, final_mask)
    print('Máscaras de união criadas!')

def make_voting_masks(images):
    final_mask = np.zeros(IMAGE_SIZE)
    
    for image in images:
        mask  = get_mask_arr(image)
        # intersecao das máscaras
        final_mask += mask

    final_mask = (final_mask >= NUM_VOTINGS)

    
    output_path = os.path.dirname(os.path.abspath(images[0]))
    image_name = os.path.basename(images[0])
    for alg in MASKS_ALGORITHMS:
        image_name = image_name.replace('_RT_{}'.format(alg), '_RT_Voting')

    output_path = os.path.join(output_path, image_name)
    write_mask(output_path, final_mask)
    print('Máscaras de votação criadas!')
        

def get_mask_arr(path):
    """ Abre a mascara como array"""
    # mask = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    # return mask

    with rasterio.open(path) as src:
        img = src.read().transpose((1, 2, 0))
        seg = np.array(img, dtype=int)

        return seg[:, :, 0], src.profile


def write_mask(mask_path, mask, profile={}):
    """Escreve as máscaras em disco"""
    # mask = np.array(mask * 255, dtype=np.uint8)
    # cv2.imwrite(mask_path, mask)

    profile.update({'dtype': rasterio.uint8,'count': 1})

    with rasterio.open(mask_path, 'w', **profile) as dst:
        dst.write_band(1, mask.astype(rasterio.uint8))



folders = os.listdir(MASKS_DIR)

for folder in folders:
    path = os.path.join(MASKS_DIR, folder)

    images = load_images(path)

    if len(images) != len(MASKS_ALGORITHMS):
        continue

    make_intersection_masks(images)
    make_union_masks(images)
    make_voting_masks(images)

print('DONE')

