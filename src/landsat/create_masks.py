# -*- coding: utf-8 -*-
#===============================================================================

import os
import math
import glob

import numpy as np
# import pandas as pd

import requests

import cv2
import gdal
# from osgeo import gdal
import rasterio



#===============================================================================
# CONSTANTS
#===============================================================================

#AWS_18 = 'http://landsat-pds.s3.amazonaws.com/c1/L8/'
GC_L8 = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/'

MTL_EXTENSION = '_MTL.txt'

# BQA_PATH = '/home/gabriel/BQA/'
#IN_DIR = r'/media/hd2/Landsat8/'
#OUT_DIR = r'/media/hd2/Landsat8/ALL_MASKS/'

IN_DIR = r'../../dataset/images/tif_images/'
OUT_DIR = r'../../dataset/masks/'


if not os.path.exists(OUT_DIR): 
    os.makedirs(OUT_DIR)             

if not os.path.exists(OUT_DIR+'patches/'): 
    os.makedirs(OUT_DIR+'patches/')             


# IN_DIR = r'C:\Users\aloga\Desktop\Mestrado\Downloads\202008\\'
# OUT_DIR = r'C:\Users\aloga\Desktop\Mestrado\Downloads\202008\\'




#===============================================================================
# FUNCTIONS
#===============================================================================

def getMTLParameters (MTL):
    '''Parses the given metadata (MTL) text, and returns several independent
parameters.'''

    Mref = []
    Aref = []
    Mrad = []
    Arad = []
    K1 = []
    K2 = []

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
        if 'FILE_DATE' in ln:
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
def get_bounds(width,height,transform):

    left = int(float(transform[2]))
    right = int(float(transform[2])) + int(float(width))*int(float(transform[0]))
    bottom = int(float(transform[5])) + int(float(height))*int(float(transform[4]))
    top = int(float(transform[5]))

    bounds = (left, bottom, right, top)

    return bounds


#-------------------------------------------------------------------------------
def get_extent(dataset):

    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    transform = dataset.GetGeoTransform()

    minx = transform[0]
    maxx = transform[0] + cols * transform[1] + rows * transform[2]
    miny = transform[3] + cols * transform[4] + rows * transform[5]
    maxy = transform[3]

    return {"minX": str(minx), "maxX": str(maxx),
            "minY": str(miny), "maxY": str(maxy),
            "cols": str(cols), "rows": str(rows)}




#-------------------------------------------------------------------------------
def getReflectance (band, add_band, mult_band, sun_elevation):
    '''A tiny function, used just to compute the reflectances, with correction
for solar angle (given in degrees).'''

    p = ((band * mult_band) + add_band) # TOA planetary reflectance, without correction for solar angle
    corrected = p / math.sin (math.radians (sun_elevation)) # TOA planetary reflectance, with correction for solar angle

    return p, corrected



#-------------------------------------------------------------------------------
def get_saturation(BQA):
    vals = [2724,2756,2804,2980,3012,3748,3780,6820,6852,6900,7076,7108,7844,7876,
            2728,2760,2808,2984,3016,3752,3784,6824,6856,6904,7080,7112,7848,7880,
            2732,2764,2812,2988,3020,3756,3788,6828,6860,6908,7084,7116,7852,7884]
    
    sat = np.zeros((BQA.shape), dtype=bool)

    for val in vals:
        sat = sat | (BQA==val)
        
    return sat.astype(np.int)

  
    



#-------------------------------------------------------------------------------
def save_masks(out_dir, image_name, profile, fire_mask, reference):
    # Save (only if fire was found!)
    if (np.amax (fire_mask) >= 1):

        profile.update ({'driver': 'GTiff',
                         'dtype': rasterio.uint8,
                         'height': fire_mask.shape[0],
                         'width': fire_mask.shape[1],
                         'count': 1})

        out_filename = os.path.join (out_dir, image_name + '_' + reference + '.TIF')

        if not os.path.exists (out_filename):
            with rasterio.open (out_filename, 'w', **profile) as dst:
                dst.write_band (1, fire_mask.astype (rasterio.uint8))

        # Save a binary mask in png.
        out_filename = os.path.join (out_dir, image_name + '_' + reference + '.png')
        cv2.imwrite (out_filename, fire_mask*255)





#-------------------------------------------------------------------------------
def get_split(fileIMG,out_path):
    
    dataset = gdal.Open(fileIMG)
    mask = dataset.GetRasterBand(1).ReadAsArray()


    passo = 256
    xsize = 1*passo
    ysize = 1*passo

    extent = get_extent(dataset)
    cols = int(extent["cols"])
    rows = int(extent["rows"])

    nx = (math.ceil(cols/passo))
    ny = (math.ceil(rows/passo))
    
    #print(nx*ny)

    cont = 0
    contp = 0

    for i in range(0,nx):
        for j in range(0,ny):
            cont += 1
            dst_dataset = out_path+os.path.basename(fileIMG)[:-4]+'_p'+str(cont).zfill(5)+'.tif'


            if not os.path.exists(dst_dataset):
                xoff = passo*i
                yoff = passo*j
                
                if xoff+xsize > cols: 
                    n2 = range(xoff,cols)
                else:
                    n2 = range(xoff,xoff+xsize)                    
                    
                if yoff+ysize > rows: 
                    n1 = range(yoff,rows) 
                else:
                    n1 = range(yoff,yoff+ysize)
                    
                    
                if np.amax(mask[np.ix_(n1,n2)]):
                    contp += 1
                    gdal.Translate(dst_dataset, dataset, srcWin = [xoff, yoff, xsize, ysize])                
            
                    

    return contp




#===============================================================================
# EQUATIONS (Schroeder)
#===============================================================================
# The following functions implement the equations in the paper.

def Seq1 (bands, r75, diff75):
    '''Eq 1 (unambiguous fires).'''
    return (np.logical_and (bands [7] > 0.5, np.logical_and (r75 > 2.5, diff75 > 0.3)))

#-------------------------------------------------------------------------------

def Seq2 (bands):
    '''Eq 2 (unambiguous fires).'''
    return (np.logical_and (bands [6] > 0.8, np.logical_and (bands [1] < 0.2, np.logical_or (bands [5] > 0.4, bands [7] < 0.1))))

#-------------------------------------------------------------------------------

def Seq3 (r75, diff75):
    '''Eq 3 (potential fires).'''
    return (np.logical_and (r75 > 1.8, diff75 > 0.17))

#-------------------------------------------------------------------------------

def Seq4and5 (bands, r75, unamb_fires, potential_fires, water):
    '''Eq 4 and 5 (contextual test for potential fires).'''

    # Means and standard deviations are computed ignoring unambiguous fires, as
    # well as water pixels.
    ignored_pixels = np.logical_or (bands [7] <= 0, np.logical_or (unamb_fires, water))
    kept_pixels = np.logical_not (ignored_pixels)

    # Reason between bands 7 and 5
    r75_ignored = r75.copy ()
    r75_ignored [ignored_pixels] = np.nan # Fire and water pixels are set to NaN.

    band7_ignored = bands [7].copy ()
    band7_ignored [ignored_pixels] = np.nan # Fire and water pixels are set to NaN.

    # Test potential fires.
    candidates = np.nonzero (potential_fires)
    for i in range (len (candidates [0])):
        y = candidates [0][i]
        x = candidates [1][i]

        # 61x61 window.
        t = max (0,y-30)
        b = min (potential_fires.shape [0], y+31)
        l = max (0, x-30)
        r = min (potential_fires.shape [1], x+31)

        eq4_result = r75 [y,x] > np.nanmean (r75_ignored [t:b,l:r]) + np.maximum (3 * (np.nanstd (r75_ignored [t:b,l:r])), 0.8)
        eq5_result = bands [7][y,x] > np.nanmean (band7_ignored [t:b,l:r]) + np.maximum (3 * (np.nanstd (band7_ignored [t:b,l:r])), 0.08)
        if not (eq4_result) or not (eq5_result):
            potential_fires [y,x] = False

    return potential_fires

#-------------------------------------------------------------------------------

def Seq6 (bands):
    '''Eq 6 (additional test for potential fires).'''
    # Avoid divisions by 0!
    p6 = np.where (bands[6] == 0, np.finfo (float).eps, bands[6])
    return (bands [7] / p6 > 1.6)

#-------------------------------------------------------------------------------

def Seq7_8_9 (bands):
    '''Eq 7, 8 and 9 (water test).'''
    result7 = np.logical_and (bands [4] > bands [5], np.logical_and (bands [5] > bands [6], np.logical_and (bands [6] > bands [7], bands [1] - bands [7] < 0.2)))
    return (np.logical_and (result7, np.logical_or (bands [3] > bands [2], np.logical_and (bands [1] > bands [2], np.logical_and (bands [2] > bands [3], bands [3] > bands [4])))))




#===============================================================================
# EQUATIONS (Kumar-Roy)
#===============================================================================
# The following functions implement the equations in the Kumar-Roy's paper.

def Geq12 (bands):
    '''Eq 12 (unambiguous fires).'''
    return (bands [4] <= 0.53 * bands [7] - 0.214)

#-------------------------------------------------------------------------------

def Geq13 (bands, eq12_mask):
    '''Eq 13 (unambiguous fires near pixels detected by eq 12).'''

    neighborhood = cv2.dilate (eq12_mask.astype (np.uint8), cv2.getStructuringElement (cv2.MORPH_RECT, (3,3))).astype (eq12_mask.dtype)
    # Striclty speaking, we should take out from the neighborhood the pixels
    # that were set in eq12, but as both eq12 and eq13 indicate unambiguous
    # fires, the end result should be the same.
    #neighborhood = np.logical_xor (neighborhood, eq12_mask)
    return (np.logical_and (neighborhood, bands [4] <= 0.35 * bands [6] - 0.044))

#-------------------------------------------------------------------------------

def Geq14 (bands):
    '''Eq 14 (potential fires).'''
    return (bands [4] <= 0.53 * bands [7] - 0.125)

#-------------------------------------------------------------------------------

def Geq15 (bands):
    '''Eq 15 (potential fires).'''
    return (bands [6] <= 1.08 * bands [7] - 0.048)

#-------------------------------------------------------------------------------

def Geq16 (bands):
    '''Eq 16 (water test).'''
    return (np.logical_and (np.logical_and (bands [2] > bands [3], bands [3] > bands [4]), bands [4] > bands [5]))

#-------------------------------------------------------------------------------

def pixelVal(p7,ef,ep,ew):
    # e = (p7>0) & (~ef) & (~ep) & (~ew)
    e = np.logical_and (p7>0, np.logical_and (np.logical_not (ef), np.logical_and (np.logical_not (ep), np.logical_not (ew))))
    return e

#-------------------------------------------------------------------------------

def Geq8and9 (bands, valid, unamb_fires, potential_fires, water):
    '''Eq 8 and 9 (contextual test for potential fires).'''

    # Means and standard deviations are computed ignoring unambiguous and
    # potential fires, as well as water and shadow pixels. The paper is not
    # clear on whether we should consider eq16 for the water pixels, or eq11
    # (from Schroeder, et al.). Eq 16 is used to define the neighborhood size,
    # so we will use it for everything here.
    ignored_pixels = np.logical_or (unamb_fires, np.logical_or (potential_fires, water))
    ignored_pixels = np.logical_or (ignored_pixels, np.logical_not (valid))
    kept_pixels = np.logical_not (ignored_pixels)

    # Reason between bands 7 and 5
    r75 = bands [7] / bands [5]
    r75_ignored = r75.copy ()
    r75_ignored [ignored_pixels] = np.nan # Fire and water pixels are set to NaN.

    band7_ignored = bands [7].copy ()
    band7_ignored [ignored_pixels] = np.nan # Fire and water pixels are set to NaN.

    # Growing region.
    sizes = list(range(5,61+2,2))

    candidates = np.nonzero (potential_fires) # Test potential fires.

    for i in range (len (candidates [0])):
        y = candidates [0][i]
        x = candidates [1][i]
        tested = False
        for w in sizes:
            t = max (0,y-w//2)
            b = min (potential_fires.shape [0], y+w//2+1)
            l = max (0, x-w//2)
            r = min (potential_fires.shape [1], x+w//2+1)

            # Stop when at least 25% of the pixels were kept (not fire or water).
            if np.count_nonzero (kept_pixels [t:b,l:r]) >= 0.25 * (b-t)*(r-l):
                tested = True
                eq8_result = r75 [y,x] > np.nanmean (r75_ignored [t:b,l:r]) + np.maximum (3 * (np.nanstd (r75_ignored [t:b,l:r])), 0.8)
                eq9_result = bands [7][y,x] > np.nanmean (band7_ignored [t:b,l:r]) + np.maximum (3 * (np.nanstd (band7_ignored [t:b,l:r])), 0.08)
                if not (eq8_result) or not (eq9_result):
                    potential_fires [y,x] = False
                break

        if not tested:
            potential_fires [y,x] = False

    return potential_fires


#===============================================================================
# EQUATIONS (MURPHY)
#===============================================================================
# The following functions implement the equations in the Murphy's paper.

def Meq2 (bands):
    '''Eq 2 (unambiguous fires).'''

    # Avoid divisions by 0!
    p5 = np.where (bands[5] == 0, np.finfo (float).eps, bands[5])
    p6 = np.where (bands[6] == 0, np.finfo (float).eps, bands[6])
    return (np.logical_and (bands[7] >= 0.15, np.logical_and (bands[7]/p6 >= 1.4, bands[7]/p5 >= 1.4)))

#-------------------------------------------------------------------------------

def Meq3 (bands, unamb, sat):
    '''Eq 3 (potential fires).'''

    neighborhood = cv2.dilate (unamb.astype (np.uint8), cv2.getStructuringElement (cv2.MORPH_RECT, (3,3))).astype (unamb.dtype)
    # Striclty speaking, we should take out from the neighborhood the pixels
    # that were set by eq 2, but the results will be joined anyway...
    #neighborhood = np.logical_xor (neighborhood, unamb)

    # Avoid divisions by 0!
    p5 = np.where (bands[5] > 0, np.finfo (float).eps, bands[5])
    return (np.logical_and (neighborhood, np.logical_or (np.logical_and (bands[6]/p5 >= 2.0, bands[6]>=0.5), sat)))



#===============================================================================
# CENTRAL FIRE DETECTION FUNCTIONS
#===============================================================================

def getFireMaskGOLI (bands):
    '''This is the central function. Receives the (corrected) reflectance bands
and returns a binary fire mask.'''

    # Exclude from every step positions with band 7 <= 0.
    valid = bands [7] > 0
    valid = cv2.erode (valid.astype (np.uint8), cv2.getStructuringElement (cv2.MORPH_RECT, (3,3))).astype (np.uint8)


    # Unambiguous fires (satisfy eq 12 or 13).
    unamb_fires = Geq12 (bands)
    unamb_fires = np.logical_and (valid, unamb_fires)
    if np.any (unamb_fires): # Run eq 13 only if needed.
        unamb_fires = np.logical_or (unamb_fires, Geq13 (bands, unamb_fires))
        unamb_fires = np.logical_and (valid, unamb_fires)

    # Potential fires (satisfy eq 14 or 15).
    potential_fires = Geq14 (bands)
    potential_fires = np.logical_or (potential_fires, Geq15 (bands))
    potential_fires = np.logical_and (valid, potential_fires)

    # Water pixels (used by the contextual test and excluded from the result.
    water = Geq16 (bands)

    # Contextual test for potential fires (eq 8 and 9).
    if np.any (potential_fires):
        potential_fires = Geq8and9 (bands, valid, unamb_fires, potential_fires, water)

    final_mask = np.logical_and (np.logical_or (unamb_fires, potential_fires), np.logical_not (water))
    return (final_mask.astype (np.int))



#-------------------------------------------------------------------------------

def getFireMaskMurphy (bands, saturated):
    '''This is the central function. Receives the (corrected) reflectance bands
and a binary mask indicating saturated pixels, and returns a binary fire mask.'''

    unamb_fires = Meq2 (bands)

    if np.any (unamb_fires): # Run eq 3 only if needed.
        potential_fires = Meq3 (bands, unamb_fires, saturated)
        final_mask = (unamb_fires | potential_fires)
    else:
        final_mask = unamb_fires

    return (final_mask.astype (np.int))


#-------------------------------------------------------------------------------
def getFireMaskSchroeder (bands):

    r75 = bands [7] / bands [5] # Compute only once (used by multiple equations).
    diff75 = bands [7] - bands [5] # Compute only once (used by multiple equations).

    # Unambiguous fires (satisfy eq 1 or 2).
    unamb_fires = Seq1 (bands, r75, diff75)
    unamb_fires = np.logical_or (unamb_fires, Seq2 (bands))

    # Potential fires (satisfy eq 3).
    potential_fires = Seq3 (r75, diff75)

    # Test eq 6 before eq 4 and 5 in an attempt to avoid the time-consuming contextual test when possible.
    potential_fires = np.logical_and (potential_fires, Seq6 (bands))

    # Water pixels (used by the contextual test and excluded from the result.
    water = Seq7_8_9 (bands)

    # Contextual test for potential fires (eq 4 and 5).
    if np.any (potential_fires):
        potential_fires = Seq4and5 (bands, r75, unamb_fires, potential_fires, water)

    final_mask = np.logical_and (np.logical_or (unamb_fires, potential_fires), np.logical_not (water))
    return (final_mask.astype (np.int))



#-------------------------------------------------------------------------------
def processImage (in_dir, out_dir, image_name, Aref, Mref, SE, sat):
    '''Reads a .tif image in the input directory, obtains metadata from AWS and
reflectance values for each band, performs fire detection, and saves the results
to other files in the output directory.

Parameters: in_dir: input directory.
            out_dir: output directory.
            image_name: image name (without extension).

Return value: none. Saves the output to a .tif file in the output directory,
    with the same name as image_name, with 'Reference' [Schroeder, GOLI, or Murphy]
    appended to the end, as well as a binary .png image with the same name,
    containing only the fire mask.'''

    # Read data from the file. The bands are numbered 1~11, but this algoritm
    # uses only bands 1~7. Bands 8~11 are set to None.
    bands = []
    
    with rasterio.open (os.path.join (in_dir, image_name + '.TIF')) as src:
        profile = src.profile.copy ()
        for i in range (12):
            if i < 1 or i > 7: # Keep bands 1 to 7.
                bands.append (None)
            else:
                bands.append (src.read (i))



    reflectance = np.copy(bands)
    corrected = np.copy(bands)
    
    # Get corrected reflectances for bands 1~7.
    for i in range (1,8):
        reflectance [i], corrected [i] = getReflectance (bands [i], Aref [i-1], Mref [i-1], SE)
    bands = None


    # Get the fire mask.
    fire_mask = getFireMaskSchroeder (reflectance)
    save_masks(out_dir, image_name, profile, fire_mask, 'Schroeder')
    if os.path.exists(out_dir + image_name + '_Schroeder.TIF'):
    	_ = get_split(out_dir + image_name + '_Schroeder.TIF',out_dir+'patches/')    
    reflectance = None


    fire_mask = getFireMaskGOLI (corrected)
    save_masks(out_dir, image_name, profile, fire_mask, 'Kumar-Roy')
    if os.path.exists(out_dir + image_name + '_' + 'Kumar-Roy' + '.TIF'):
        _ = get_split(out_dir + image_name + '_' + 'Kumar-Roy' + '.TIF',out_dir+'patches/')

    fire_mask = getFireMaskMurphy (corrected,sat)
    save_masks(out_dir, image_name, profile, fire_mask, 'Murphy')
    if os.path.exists(out_dir + image_name + '_' + 'Murphy' + '.TIF'):
        _ = get_split(out_dir + image_name + '_' + 'Murphy' + '.TIF',out_dir+'patches/')
    corrected = None






#===============================================================================
# TEST SCRIPT
#===============================================================================

files = glob.glob(IN_DIR + '*BQA.TIF')
print('Files to process:', len(files))
files.reverse()

for ind, file in enumerate(files):
    try:
        image_name = os.path.basename(file.replace('_BQA.TIF',''))
        print(ind,'-',image_name)
    
#------ BQA - SATURATION BAND -----------------------------------------------            
        with rasterio.open(file) as src:
            profile = src.profile.copy()
            BQA = src.read(1)
                
        sat = get_saturation(BQA)
                    
            
#------ MTL -----------------------------------------------
        aws_path = GC_L8 + image_name [10:13] + '/' + image_name [13:16] + '/' + image_name + '/' + image_name + MTL_EXTENSION
        MTL = requests.get(aws_path).text
        Mrad, Arad, Mref, Aref, K1, K2, SE, L8ID, FDATE, DATEAC, SceneTIME, CC, MP, DT, EL, ZONE = getMTLParameters (MTL)


#------ Process -----------------------------------------------
        if os.path.exists(os.path.join(OUT_DIR,image_name + '_Murphy.TIF')) or os.path.exists(os.path.join(OUT_DIR,image_name + '_GOLI.TIF')) or os.path.exists(os.path.join(OUT_DIR,image_name + '_Schroeder.TIF')):
            print('    ja processado')
        else:
            processImage (IN_DIR, OUT_DIR, image_name, Aref, Mref, SE, sat)

    except Exception as e:
        print(e)

