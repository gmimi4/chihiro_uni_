# -*- coding: utf-8 -*-
"""
# extract mean pearson for validation
"""

import os
import glob
import rasterio
import numpy as np
from tqdm import tqdm
import re

# in_dir_parent = r"D:\Malaysia\02_Timeseries\CPA_CPR\6_time_lag\EVI"
in_dir_parent = r"D:\Malaysia\02_Timeseries\CPA_CPR\6_time_lag"
out_dir = in_dir_parent + os.sep + "lag_mean"
os.makedirs(out_dir, exist_ok=True)

dirs = os.listdir(in_dir_parent)
dirs = [d for d in dirs if d.startswith("lag_")]
# dirs = [d for d in dirs if isinstance(int(d[4:5]),int)]
dirs = [d for d in dirs if (not "mean" in d) and (not "best" in d) ]
lagnum_list = [n[-1] for n in dirs]

variable_list = [
                   # "GOSIF",
                   "rain", 
                   "temp",
                   "VPD",
                   "Et",
                   "Eb",
                   "SM",
                  "VOD"
                 ]


periods = ["2002-2022"] #"2000-2023" "2002-2012","2013-2022",
     
for pagenum in ["A1","A2","A3","A4"]:
    for p in ["pearson","pval"]:
        for peri in periods:
            for variable in tqdm(variable_list):
                print(f"processing {p} for {peri}")
            
                tifs = []
                for d in dirs: #lag_t    
                    in_dir = os.path.join(in_dir_parent, d)
                    tif = glob.glob(in_dir + os.sep + f"*{pagenum}_{variable}*{p}*{peri}*.tif")[0]
                    
                    # lagnum = d[-1]
                    tifs.append(tif)
                    # tifs.append(tif)
                    
                arr_list = []
                for t in tifs:
                    src = rasterio.open(t) #do not close src
                    arr = src.read(1)
                    arr_1d = np.ravel(arr)
                    arr_list.append(arr_1d)
                    
                    height = arr.shape[0]
                    width = arr.shape[1]
                    out_meta = src.meta.copy()
                    src.close()
                
                arr_stack = np.stack(arr_list)
                arr_stack_mean = np.nanmean(arr_stack, axis=0) #列ごと
                    
                # reshape
                arr_result_re = arr_stack_mean.reshape((height, width))
                
                # out_meta.update({'dtype':'uint8','nodata':0 })   
                
                filename = os.path.basename(tifs[0])
                lagn = filename.split("_")[3]
                outfilename = filename.replace(lagn,"mean")
                outfile = out_dir + os.sep + outfilename
                with rasterio.open(outfile, 'w', **out_meta) as dst:
                    dst.write(arr_result_re,1)


# import glob
# tmp_tifs = glob.glob(out_dir + os.sep + "*.tif")
# for t in tmp_tifs:
#     os.remove(t)
