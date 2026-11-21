# -*- coding: utf-8 -*-
#===============================================================================
# Creates the active fire masks (Schroeder et al., 2016; Kumar and Roy, 2018;
# Murphy et al., 2016) from Landsat-8/9 Level-1 scenes.
#
#   python create_masks.py --collection c2 [--in_dir DIR] [--out_dir DIR]
#   python create_masks.py --collection c1 [--in_dir DIR] [--out_dir DIR]
#
# c1: Collection 1, the input used to create the published dataset (stacked
#     <scene>.TIF + <scene>_BQA.TIF, MTL from Google Cloud). USGS removed the
#     Collection 1 products in December 2022.
# c2: Collection 2, one folder per scene as delivered by USGS (<scene>_B1.TIF
#     ... _B7.TIF, <scene>_MTL.txt and <scene>_QA_RADSAT.TIF).
# The readers are in inputs.py; the fire detection functions below are the
# same for both collections.
#===============================================================================

import os
import math
import glob
import argparse

import numpy as np
# import pandas as pd

import cv2
from osgeo import gdal
import rasterio

from inputs import read_c1, read_c2, list_c2_scenes



#===============================================================================
# CONSTANTS
#===============================================================================

# BQA_PATH = '/home/gabriel/BQA/'
#IN_DIR = r'/media/hd2/Landsat8/'
#OUT_DIR = r'/media/hd2/Landsat8/ALL_MASKS/'

IN_DIR = r'../../dataset/images/tif_images/'     # Collection 1 (default of --in_dir with --collection c1)
IN_DIR_C2 = r'../../dataset/images/c2/'          # Collection 2 (default of --in_dir with --collection c2)
OUT_DIR = r'../../dataset/masks/'


# IN_DIR = r'C:\Users\aloga\Desktop\Mestrado\Downloads\202008\\'
# OUT_DIR = r'C:\Users\aloga\Desktop\Mestrado\Downloads\202008\\'




#===============================================================================
# FUNCTIONS
#===============================================================================

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
    p5 = np.where (bands[5] == 0, np.finfo (float).eps, bands[5])
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
    return (final_mask.astype (int))



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

    return (final_mask.astype (int))


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
    return (final_mask.astype (int))



#-------------------------------------------------------------------------------
def processImage (out_dir, image_name, profile, reflectance, corrected, sat):
    '''Performs fire detection on the reflectance bands returned by one of the
readers in inputs.py (read_c1 or read_c2), and saves the results to other files
in the output directory.

Parameters: out_dir: output directory.
            image_name: image name (without extension).
            profile: rasterio profile of the input image.
            reflectance: TOA reflectance without correction for solar angle
                (list indexed by band number, bands 1~7).
            corrected: TOA reflectance with correction for solar angle.
            sat: saturation mask (used by Murphy et al., 2016).

Return value: none. Saves the output to a .tif file in the output directory,
    with the same name as image_name, with 'Reference' [Schroeder, Kumar-Roy, or Murphy]
    appended to the end, as well as a binary .png image with the same name,
    containing only the fire mask.'''

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
# SCRIPT
#===============================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser (description='Creates the active fire masks from Landsat-8/9 Level-1 scenes.')
    parser.add_argument ('--collection', choices=['c1', 'c2'], default='c2',
                         help='c1: Collection 1 (input of the published dataset); c2: Collection 2 (default).')
    parser.add_argument ('--in_dir', default=None,
                         help='Input folder (default: %s for c1, %s for c2).' % (IN_DIR, IN_DIR_C2))
    parser.add_argument ('--out_dir', default=OUT_DIR, help='Output folder (default: %(default)s).')
    args = parser.parse_args ()

    in_dir = args.in_dir or (IN_DIR if args.collection == 'c1' else IN_DIR_C2)
    out_dir = os.path.join (args.out_dir, '')

    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)             

    if not os.path.exists(out_dir+'patches/'): 
        os.makedirs(out_dir+'patches/')             

    # (scene folder, image name) for each scene.
    if args.collection == 'c1':
        files = glob.glob(os.path.join(in_dir, '*BQA.TIF'))
        files.reverse()
        scenes = [(in_dir, os.path.basename(f.replace('_BQA.TIF',''))) for f in files]
    else:
        scenes = list_c2_scenes(in_dir)
    print('Files to process:', len(scenes))

    for ind, (scene_dir, image_name) in enumerate(scenes):
        try:
            print(ind,'-',image_name)

            if os.path.exists(os.path.join(out_dir,image_name + '_Murphy.TIF')) or os.path.exists(os.path.join(out_dir,image_name + '_Kumar-Roy.TIF')) or os.path.exists(os.path.join(out_dir,image_name + '_Schroeder.TIF')):
                print('    ja processado')
                continue

            if args.collection == 'c1':
                reflectance, corrected, sat, profile = read_c1 (scene_dir, image_name)
            else:
                reflectance, corrected, sat, profile = read_c2 (scene_dir, image_name)

            processImage (out_dir, image_name, profile, reflectance, corrected, sat)

        except Exception as e:
            print(e)
