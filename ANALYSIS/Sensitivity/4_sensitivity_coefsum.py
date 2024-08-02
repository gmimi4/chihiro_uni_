# -*- coding: utf-8 -*-
"""
# change to cv
# 
"""

import numpy as np
import pandas as pd
import os,sys
import glob
import rasterio
from tqdm import tqdm
import copy


in_dir_importane = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_01"
# p_val_tif = f'/Volumes/SSD_2/Malaysia/02_Timeseries/CPA_CPR/2_out_ras/p_01/{PageName}_p_values_importance_2013-2022.tif' #as sample tif
out_dir = r'D:\Malaysia\02_Timeseries\Sensitivity\1_sum'
os.makedirs(out_dir,exist_ok=True)
coef_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_01"

pages = ["A1","A2","A3","A4"]
periods = ["2002-2012","2013-2022","2002-2022"]

var_list = ["rain",
            "temp",
            "Eb",
            "Et",
            "VPD",
            "SM",
            "VOD"]

""" # simply sum all coef"""
        
for page in pages:
    for peri in periods:
        importance_tifs = glob.glob(in_dir_importane + os.sep + f"*{peri}.tif")
        importance_tifs = [t for t in importance_tifs if os.path.basename(t)[:-4].split("_")[1] in var_list]
        use_tifs = [t for t in importance_tifs if (os.path.basename(t)[:-4].split("_")[0] == page)]
        
        arr_list = []
        for tif in use_tifs:
            with rasterio.open(tif) as src:
                meta =src.meta
                arr = src.read(1)
                arr_abs = abs(arr)
            
            arr_list.append(arr_abs)
        
        arr_stck = np.stack(arr_list)
        arr_sum = np.sum(arr_stck, axis=0)
        
    
        """　# tifのExport """
        outfile = os.path.join(out_dir,f"{page}_sensitivity_sumcoef_{peri}.tif")
        with rasterio.open(outfile, "w", **meta) as dst:
            dst.write(arr_sum, 1)








