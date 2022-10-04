# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 11:20:15 2020

@author: aloga
"""

#%%
#===============================================================================
import os
import time

import pandas as pd
import rasterio

#%%
#===============================================================================

#AWS_L8 = 'http://landsat-pds.s3.amazonaws.com/c1/L8/'
GC_L8 = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/'

MTL_EXTENSION = '_MTL.txt'
ext = '.TIF'

IN_DIR = r'../../dataset/'
OUT_DIR = r'../../dataset/images/tif_images/'

if not os.path.exists(OUT_DIR): 
        os.makedirs(OUT_DIR)             


#%%
#===============================================================================
csv = "images202009.csv"
df = pd.read_csv(os.path.join(IN_DIR, csv), sep=';')


#%%
#===============================================================================
for ind,image_name in enumerate(df.productId.to_list()):
    
    start_time = time.time()
    print(ind+1,image_name)
    
    log = OUT_DIR + image_name + '.log'  
    if not os.path.exists(log):
        #try:               
            path  = image_name[10:13]
            row = image_name[13:16]            
            
            ########################### BQA ################################
            outnameBQA = OUT_DIR + image_name + '_' + 'BQA' + ext 
            if not os.path.exists(outnameBQA):
                link = GC_L8 + path + '/' + row + '/' + image_name + '/' + image_name + '_' + 'BQA' + ext
                with rasterio.open(link) as src:
                    profile = src.profile.copy()
                    BQA = src.read(1)
                
                with rasterio.open(outnameBQA, 'w', **profile) as dst:
                    dst.write_band(1, BQA.astype(rasterio.uint16))
                    
            
            ########################### BANDS ################################
            outname = OUT_DIR + image_name + ext   # checar se já foi feito download
            if not os.path.exists(outname):
                link_bands = [GC_L8 + path + '/' + row + '/' + image_name + '/' + image_name + '_' + 'B1' + ext, 
                            GC_L8 + path + '/' + row + '/' + image_name + '/' + image_name + '_' + 'B2' + ext, 
                            GC_L8 + path + '/' + row + '/' + image_name + '/' + image_name + '_' + 'B3' + ext, 
                            GC_L8 + path + '/' + row + '/' + image_name + '/' + image_name + '_' + 'B4' + ext, 
                            GC_L8 + path + '/' + row + '/' + image_name + '/' + image_name + '_' + 'B5' + ext, 
                            GC_L8 + path + '/' + row + '/' + image_name + '/' + image_name + '_' + 'B6' + ext, 
                            GC_L8 + path + '/' + row + '/' + image_name + '/' + image_name + '_' + 'B7' + ext,
                            GC_L8 + path + '/' + row + '/' + image_name + '/' + image_name + '_' + 'B9' + ext, 
                            GC_L8 + path + '/' + row + '/' + image_name + '/' + image_name + '_' + 'B10' + ext, 
                            GC_L8 + path + '/' + row + '/' + image_name + '/' + image_name + '_' + 'B11' + ext]
            
                bands = []
                for link in link_bands:
                    with rasterio.open(link) as src:
                        bands.append(src.read(1))
                        
                        
                profile.update({'dtype': rasterio.uint16,
                                    'height': bands[0].shape[0],
                                    'width': bands[0].shape[1],
                                    'count': len(bands)})
                
                with rasterio.open(outname, 'w', **profile) as dst:
                    for i,band in enumerate(bands):
                        dst.write_band(i+1, band.astype(rasterio.uint16))


            
            f = open(log, 'w+')                
            f.write('Done!')
            f.close()

            end_time = time.time ()    
            print ('Elapsed: ' + str (round ((end_time-start_time)/60, 2)) + ' min.\n')
            
"""         except Exception as e:
            f = open(log, 'w+')                
            f.write(str(e))
            f.close()
            end_time = time.time ()
            print('Erro. Elapsed: ' + str (round ((end_time-start_time)/60, 2)) + ' min.\n') """
            
    

