# -*- coding: utf-8 -*-
"""
#extract forest pixels
"""
import rasterio
import numpy as np
import os

tif = r"F:\MAlaysia\MODIS_IGBP\MODIS_IGBP_mosaic.tif"
out_dir = os.path.dirname(tif)

with rasterio.open(tif) as src:
    meta = src.meta
    arr = src.read(1)
    arr_forest = np.where((arr ==1)|(arr ==2)|(arr ==3)|(arr ==4)|(arr ==5),1,0)
    
    outfile = out_dir + os.sep + os.path.basename(tif)[:-4] + "_forest.tif"
    with rasterio.open(outfile, 'w', **meta) as dst:
        dst.write(arr_forest, 1)
