# -*- coding: utf-8 -*-
"""
#extract forest pixels
"""
import rasterio
import numpy as np
import os

tif = r"F:\MAlaysia\MODIS_IGBP\MODIS_IGBP_mosaic.tif"
out_dir = os.path.dirname(tif)

reclass_map = {
    1: 40, #forest
    2: 40,
    3: 40,
    4: 40,
    5: 40,
    6: 30, #shrub and grass
    7: 30,
    8: 30,
    9: 30,
    10:30,
    11:99, #others
    12:20, #cropland
    13:99,
    14:20,
    15:99,
    16:99,
    17:99,  
    }

with rasterio.open(tif) as src:
    meta = src.meta
    arr = src.read(1)
    meta = src.meta
    
    result = np.copy(arr)
    
    for old_value, new_value in reclass_map.items():
        result[arr == old_value] = new_value
        
    
    outfile = out_dir + os.sep + os.path.basename(tif)[:-4] + "_reclass.tif"
    with rasterio.open(outfile, 'w', **meta) as dst:
        dst.write(result, 1)
