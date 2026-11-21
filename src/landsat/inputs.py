# -*- coding: utf-8 -*-
'''Input readers used by create_masks.py.

Both readers return the same values, so the fire detection functions in
create_masks.py do not depend on the Landsat collection:

    reflectance, corrected, saturated, profile

    reflectance: list indexed by band number (0~11). Bands 1~7 hold the TOA
                 reflectance WITHOUT correction for the solar angle (used by
                 Schroeder et al., 2016). The other positions are None.
    corrected:   same, WITH correction for the solar angle (used by Kumar and
                 Roy, 2018, and Murphy et al., 2016).
    saturated:   boolean mask of saturated pixels (used by Murphy et al., 2016).
    profile:     rasterio profile used to save the masks.

Collection 1 (read_c1) is the reader used to create the published dataset.
USGS removed the Collection 1 products in December 2022, so it is kept for
reference. Collection 2 (read_c2) reads a Level-1 scene folder as delivered by
USGS (one GeoTIFF per band, the MTL.txt file and the QA_RADSAT band). It works
with Landsat-8 and Landsat-9 scenes.
'''

import os
import math
import glob

import numpy as np
import requests
import rasterio



#===============================================================================
# CONSTANTS
#===============================================================================

#AWS_18 = 'http://landsat-pds.s3.amazonaws.com/c1/L8/'
GC_L8 = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/'

MTL_EXTENSION = '_MTL.txt'

# Collection 2 QA_RADSAT: bit 0~6 = bands 1~7, bit 8~10 = bands 9~11.
RADSAT_BIT = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 9: 8, 10: 9, 11: 10}



#===============================================================================
# FUNCTIONS (moved from create_masks.py)
#===============================================================================

def getMTLParameters (MTL):
    '''Parses the given metadata (MTL) text, and returns several independent
parameters. Works with Collection 1 and Collection 2 MTL files.'''

    Mref = []
    Aref = []
    Mrad = []
    Arad = []
    K1 = []
    K2 = []

    # Not every key exists in both collections (e.g. FILE_DATE is
    # DATE_PRODUCT_GENERATED in Collection 2).
    SE = L8ID = FDATE = DATEAC = SceneTIME = CC = MP = DT = EL = ZONE = None

    MTL = MTL.splitlines()

    for ln in MTL:

        if 'RADIANCE_MULT_BAND_' in ln:
            Mrad.append(float(ln.split(' = ')[1]))
        if 'RADIANCE_ADD_BAND_' in ln:
            Arad.append(float(ln.split(' = ')[1]))
        if 'REFLECTANCE_MULT_BAND_' in ln:
            Mref.append(float(ln.split(' = ')[1]))
        if 'REFLECTANCE_ADD_BAND_' in ln:
            Aref.append(float(ln.split(' = ')[1]))
        if 'K1_CONSTANT_BAND_' in ln:
            K1.append(float(ln.split(' = ')[1]))
        if 'K2_CONSTANT_BAND_' in ln:
            K2.append(float(ln.split(' = ')[1]))

        if 'SUN_ELEVATION' in ln:
            SE = float(ln.split(' = ')[1])

        if 'LANDSAT_SCENE_ID' in ln:
            L8ID = (ln.split(' = ')[1])
        if 'FILE_DATE' in ln or 'DATE_PRODUCT_GENERATED' in ln:
            FDATE = str(ln.split(' = ')[1])
        if 'DATE_ACQUIRED' in ln:
            DATEAC = str(ln.split(' = ')[1])
        if 'SCENE_CENTER_TIME' in ln:
            SceneTIME = str(ln.split(' = ')[1])
        if 'CLOUD_COVER' in ln:
            CC = float(ln.split(' = ')[1])
        if 'MAP_PROJECTION' in ln:
            MP = str(ln.split(' = ')[1])
        if 'DATUM' in ln:
            DT = str(ln.split(' = ')[1])
        if 'ELLIPSOID' in ln:
            EL = str(ln.split(' = ')[1])
        if 'UTM_ZONE' in ln:
            ZONE = int(ln.split(' = ')[1])

    return Mrad, Arad, Mref, Aref, K1, K2, SE, L8ID, FDATE, DATEAC, SceneTIME, CC, MP, DT, EL, ZONE



#-------------------------------------------------------------------------------
def getReflectance (band, add_band, mult_band, sun_elevation):
    '''A tiny function, used just to compute the reflectances, with correction
for solar angle (given in degrees).'''

    p = ((band * mult_band) + add_band) # TOA planetary reflectance, without correction for solar angle
    corrected = p / math.sin (math.radians (sun_elevation)) # TOA planetary reflectance, with correction for solar angle

    return p, corrected



#-------------------------------------------------------------------------------
def get_saturation(BQA):
    '''Collection 1: saturation from the BQA codes used to create the published
dataset. Bits 2-3 of the BQA give the NUMBER of saturated bands (in any band),
not which bands are saturated.'''
    vals = [2724,2756,2804,2980,3012,3748,3780,6820,6852,6900,7076,7108,7844,7876,
            2728,2760,2808,2984,3016,3752,3784,6824,6856,6904,7080,7112,7848,7880,
            2732,2764,2812,2988,3020,3756,3788,6828,6860,6908,7084,7116,7852,7884]

    sat = np.zeros((BQA.shape), dtype=bool)

    for val in vals:
        sat = sat | (BQA==val)

    return sat.astype(int)



#===============================================================================
# NEW FUNCTIONS (Collection 2)
#===============================================================================

def get_saturation_c2(QA_RADSAT, bands=(6, 7)):
    '''Collection 2: True where any of the given bands is saturated. Murphy et al.
(2016) use the saturation of bands 6 and 7.'''
    bits = 0
    for b in bands:
        bits |= 1 << RADSAT_BIT[b]
    return (QA_RADSAT.astype(np.uint16) & bits) != 0



#-------------------------------------------------------------------------------
def list_c2_scenes(in_dir):
    '''Returns (scene_dir, image_name) for every Collection 2 scene found in
in_dir (the folder of one scene, or a folder with one sub-folder per scene).'''
    mtl_files = sorted(glob.glob(os.path.join(in_dir, '*' + MTL_EXTENSION)) +
                       glob.glob(os.path.join(in_dir, '*', '*' + MTL_EXTENSION)))
    return [(os.path.dirname(f), os.path.basename(f)[:-len(MTL_EXTENSION)]) for f in mtl_files]



#===============================================================================
# READERS
#===============================================================================

def read_c1 (in_dir, image_name):
    '''Collection 1 (original code): reads the stacked image <image_name>.TIF
(bands 1~7, 9~11) and <image_name>_BQA.TIF from in_dir, and the MTL from Google
Cloud.'''

    #------ BQA - SATURATION BAND -----------------------------------------------
    with rasterio.open(os.path.join(in_dir, image_name + '_BQA.TIF')) as src:
        BQA = src.read(1)
    sat = get_saturation(BQA)

    #------ MTL -----------------------------------------------
    aws_path = GC_L8 + image_name [10:13] + '/' + image_name [13:16] + '/' + image_name + '/' + image_name + MTL_EXTENSION
    MTL = requests.get(aws_path).text
    Mrad, Arad, Mref, Aref, K1, K2, SE, L8ID, FDATE, DATEAC, SceneTIME, CC, MP, DT, EL, ZONE = getMTLParameters (MTL)

    #------ BANDS -----------------------------------------------
    # The bands are numbered 1~11, but the algorithms use only bands 1~7.
    bands = [None] * 12
    with rasterio.open (os.path.join (in_dir, image_name + '.TIF')) as src:
        profile = src.profile.copy ()
        for i in range (1, 8):
            bands [i] = src.read (i)

    return _to_reflectance (bands, Aref, Mref, SE) + (sat, profile)



#-------------------------------------------------------------------------------
def read_c2 (scene_dir, image_name):
    '''Collection 2: reads <image_name>_B1.TIF ... _B7.TIF, <image_name>_MTL.txt
and <image_name>_QA_RADSAT.TIF from scene_dir.'''

    #------ MTL -----------------------------------------------
    with open (os.path.join (scene_dir, image_name + MTL_EXTENSION)) as f:
        MTL = f.read ()
    Mrad, Arad, Mref, Aref, K1, K2, SE, L8ID, FDATE, DATEAC, SceneTIME, CC, MP, DT, EL, ZONE = getMTLParameters (MTL)

    #------ QA_RADSAT - SATURATION BAND -----------------------------------------
    with rasterio.open (os.path.join (scene_dir, image_name + '_QA_RADSAT.TIF')) as src:
        sat = get_saturation_c2 (src.read (1))

    #------ BANDS -----------------------------------------------
    bands = [None] * 12
    for i in range (1, 8):
        with rasterio.open (os.path.join (scene_dir, image_name + '_B' + str (i) + '.TIF')) as src:
            if i == 7:
                profile = src.profile.copy ()
            bands [i] = src.read (1)

    return _to_reflectance (bands, Aref, Mref, SE) + (sat, profile)



#-------------------------------------------------------------------------------
def _to_reflectance (bands, Aref, Mref, SE):
    '''TOA reflectance for bands 1~7, without and with correction for solar angle.'''
    reflectance = [None] * 12
    corrected = [None] * 12
    for i in range (1, 8):
        reflectance [i], corrected [i] = getReflectance (bands [i], Aref [i-1], Mref [i-1], SE)
    return reflectance, corrected
