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

in_dir = '/Volumes/PortableSSD/Malaysia/ENSO/01_deviations/_allevents'
out_dir = in_dir + os.sep + '_mosaic'

variable_list = [
                  # "GOSIF",
                  "EVI",
                  "rain", 
                  "temp",
                  "VPD",
                  "Et",
                  "Eb",
                  "SM",
                 "VOD"
                 ]

""" #EVI """
for variable in variable_list:
    tifs = glob.glob(in_dir + os.sep + f"*_mindevi_{variable}.tif")
    srcs =[]
    for t in tifs:
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
    
    outfilename = os.path.basename(tifs[0])[3:]
    outfile = out_dir + os.sep + f"mosaic_{outfilename}"
    with rasterio.open(outfile, 'w', **out_meta) as dst:
        dst.write(mosaic)
        
        

for variable in variable_list:
    tifs = glob.glob(in_dir + os.sep + f"*_minmon_{variable}.tif")
    srcs =[]
    for t in tifs:
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
    
    outfilename = os.path.basename(tifs[0])[3:]
    outfile = out_dir + os.sep + f"mosaic_{outfilename}"
    with rasterio.open(outfile, 'w', **out_meta) as dst:
        dst.write(mosaic)



""" #GOSIF
period_list = ["OND","JFM","AMJ","JAS"]

for variable in variable_list:
    tifs = glob.glob(in_dir + os.sep + f"*{variable}*.tif")
    for peri in period_list:
        tifs_p = [t for t in tifs if peri in t]
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
"""
        
    
    


        
       
    
        





