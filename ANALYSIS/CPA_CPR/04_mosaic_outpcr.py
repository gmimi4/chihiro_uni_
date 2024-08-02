# -*- coding: utf-8 -*-
"""
PCA用にピクセルごとに目的変数と変数を時系列に並べたcsvを出力する。
陸域のrows*colmnsの数だけ出力される

@author: chihiro

"""
#####
#8 days meanができた後
#####

import os
import glob
import rasterio
from rasterio.merge import merge
import numpy as np
from tqdm import tqdm

in_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_01"
out_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_01\_mosaic"

variable_list = [
                  "GOSIF",
                  "rain", 
                  "temp",
                  "VPD",
                  "Et",
                  "Eb",
                  "SM",
                 "VOD"
                 ]

for variable in variable_list:
    tifs = glob.glob(in_dir + os.sep + f"*{variable}*.tif")
    periods = [os.path.basename(t[:-4].split("_")[-1]) for t in tifs]
    periods = list(set(periods))
    for p in periods:
        tifs_p = [t for t in tifs if p in t]
        srcs =[]
        for t in tifs_p:
            src = rasterio.open(t) #do not close src
            srcs.append(src)
        mosaic, out_trans = merge(srcs)
        # Update the metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": src.crs
        })
        
        outfilename = os.path.basename(tifs_p[0])[3:]
        outfile = out_dir + os.sep + f"mosaic_{outfilename}"
        with rasterio.open(outfile, 'w', **out_meta) as dst:
            dst.write(mosaic)
        
    
    


        
       
    
        





