# -*- coding: utf-8 -*-
"""
# Extract largest stressed pixels.

"""

import os
import glob
import rasterio
import numpy as np
import pandas as pd

in_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\4_stressed_pixels"
enso_tif = r"F:\MAlaysia\ENSO\01_deviations\lag0\_allseasons\combine_devi_GOSIF_min.tif"
out_dir = in_dir

tifs = glob.glob(in_dir + os.sep + "*morethan*.tif")
tif = [t for t in tifs if "morethan1" in t][0] ## morethan1 tif

def raster_array(tif_path): #2d
    with rasterio.open(tif_path) as src_def:
        arr_def = src_def.read(1)  
    return arr_def


""" # sample tif"""
with rasterio.open(tifs[0]) as src:
    meta = src.meta
    arr_sample = src.read(1)
    height, width = arr_sample.shape[0],arr_sample.shape[1]
    # meta.update({'dtype':'uint8','nodata':0 })
        

tif_name = os.path.basename(tif)[:-4]
num_change = os.path.basename(tif)[:-4][-1]

arr = raster_array(tif)
arr_1d = np.ravel(arr)

## find 75% above
arr_1d_nan = arr_1d[~np.isnan(arr_1d)]
arr_upper = np.percentile(arr_1d_nan, 75)

## find vales above upper
arr_index = np.where(arr > arr_upper)
arr_index_1d = np.where(arr_1d_nan > arr_upper)
arr_index_1d = [a for a in arr_index_1d[0]]


""" # put large value in pixels """
arr_result = np.zeros((arr.shape)) # 0 array
arr_result[:,:] = np.nan #nan array
arr_result[arr_index] = arr[arr_index]
# arr_vulnerable[ind] = arr_ratio[ind]


outfile_large = out_dir + os.sep + f"largechange_{num_change}.tif"

with rasterio.open(outfile_large, 'w', **meta) as dst:
    dst.write(arr_result, 1)
    
    
    
""" # extract enso decline pixels """
##なぜかできなかったので出力されたtifを使う
arr_large = raster_array(outfile_large)
arr_large_1d = np.ravel(arr_large)
arr_index_large = np.where(arr_large_1d>0)[0]

## enso tif
arr_enso = raster_array(enso_tif)
arr_enso_1d = np.ravel(arr_enso)

## find valid idx of ENSO decrease (not nan)
arr_enso_valididx = np.where(arr_enso_1d < 0)
arr_enso_valididx = [a for a in arr_enso_valididx[0]] # to list
# arr_enso_valididx = np.where(arr_enso < 0)
# arr_enso_valididx_tupple = []
# for i in range(len(arr_enso_valididx[0])):
#     r = arr_enso_valididx[0][i]
#     c = arr_enso_valididx[1][i]
#     arr_enso_valididx_tupple.append((r,c))

## find both index
idx_large_enso = list(set(arr_index_large)&set(arr_enso_valididx))
# idx_large_enso = [i for i in arr_enso_valididx if i in arr_index_1d ]

# back to list
# idx_large_enso = [list(data) for data in idx_large_enso]

arr_result = np.zeros(arr_1d.shape) # 0 array
arr_result[:] = np.nan #nan array
# arr_result[idx_large_enso] = arr_1d[idx_large_enso]
arr_result[idx_large_enso] = arr_1d[idx_large_enso]

# test
# test = [t[0] for t in idx_large_enso]
# np.max(test)

arr_result_re = np.reshape(arr_result, [arr.shape[0],arr.shape[1]])


outfile = out_dir + os.sep + f"largechange_enso_{num_change}.tif"

with rasterio.open(outfile, 'w', **meta) as dst:
    dst.write(arr_result_re, 1)
    
    
