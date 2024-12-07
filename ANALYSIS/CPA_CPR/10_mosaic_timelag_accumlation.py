# -*- coding: utf-8 -*-
"""
"""

import os
import glob
import rasterio
from rasterio.merge import merge
from tqdm import tqdm

in_dir_parent = r"D:\Malaysia\02_Timeseries\CPA_CPR\6_time_lag\EVI"
# in_dir_parent = r"D:\Malaysia\02_Timeseries\CPA_CPR\6_time_lag"
# in_dir_parent = r"D:\Malaysia\02_Timeseries\CPA_CPR\7_accumulation"
dirs = os.listdir(in_dir_parent)
dirs = [d for d in dirs if not "best" in d]
# lag_nums = os.listdir(in_dir_parent)
# lag_nums = [n[-1] for n in lag_nums if not "best" in n]

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

for di in tqdm(dirs):
    in_dir = os.path.join(in_dir_parent, di)
    out_dir = os.path.join(in_dir_parent,di,"_mosaic")
    os.makedirs(out_dir,exist_ok=True)
    
    for variable in variable_list:
        for p in ["pearson","pval"]:
            tifs = glob.glob(in_dir + os.sep + f"*{variable}*{p}*.tif")
            periods = [os.path.basename(t[:-4].split("_")[-1]) for t in tifs]
            periods = list(set(periods))
            for peri in periods:
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

