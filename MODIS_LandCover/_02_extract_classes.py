# -*- coding: utf-8 -*-
"""
# reclassify raster array
"""
import rasterio
import numpy as np
import os

# tif = r"F:\MAlaysia\MODIS_IGBP\ISMN_grids\IGBP_1019898_SNOTEL_to2019.tif.tif"
# out_dir = os.path.dirname(tif)

""" #Class
10:grassland
20:cropland
30:shrub
41:forest 常緑
42:forest 落葉
43:forest 混合
44:forest サバンナ
50:water
60:barren
70:snow
99:others

"""


def main(tif):
    reclass_map = {
        1: 41, #evergreen forest
        2: 41,
        3: 42, #deciduous
        4: 42,
        5: 43, #mix
        6: 30, #shrub
        7: 30,
        8: 44,
        9: 44,
        10:10, #Grasslands
        11:50, #wetland
        12:20, #cropland
        13:99, #others(cities)
        14:30,
        15:70, 
        16:60, #barren
        17:50,   
        }
    
    with rasterio.open(tif) as src:
        meta = src.meta
        arr = src.read(1)
        meta = src.meta
        
        result = np.copy(arr)
        
        for old_value, new_value in reclass_map.items():
            result[arr == old_value] = new_value
                    
        # outfile = out_dir + os.sep + os.path.basename(tif)[:-4] + "_reclass.tif"
        # with rasterio.open(outfile, 'w', **meta) as dst:
        #     dst.write(result, 1)
        
    return result



