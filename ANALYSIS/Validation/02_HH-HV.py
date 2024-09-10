# -*- coding: utf-8 -*-
"""
Compare Yield data and SAR back scattered intensity by provinces or states
"""
import os
import numpy as np
import rasterio
from tqdm import tqdm
import glob


in_tif_dir = r"D:\Malaysia\Validation\2_PALSAR_mosaic\01_res100m"
out_dir = r"D:\Malaysia\Validation\2_PALSAR_mosaic\02_HH-HV_res100m"
tifs_hh = glob.glob(in_tif_dir + os.sep + "*_HH_*.tif")
tifs_hv = glob.glob(in_tif_dir + os.sep + "*_HV_*.tif")

grids = [os.path.basename(g).split("_")[3] for g in tifs_hh]

for g in tqdm(grids):
    tif_hhs = [t for t in tifs_hh if f"_{g}_res" in t]
    tif_hvs = [t for t in tifs_hv if f"_{g}_res" in t]
    
    yearlist = [os.path.basename(t).split("_")[2] for t in tif_hhs]
    
    for yr in yearlist:
        hh = [t for t in tif_hhs if f"_{yr}_{g}_res" in t][0]
        hv = [t for t in tif_hvs if f"_{yr}_{g}_res" in t][0]
        with rasterio.open(hh) as src:
            arr_hh = src.read(1)
            profile = src.profile
        with rasterio.open(hv) as src:
            arr_hv = src.read(1)
        arr_diff = arr_hh - arr_hv
        
        outname = os.path.basename(hh)[:-4].replace("HH","HH-HV") + ".tif"
        outfile = out_dir + os.sep + outname
        with rasterio.open(outfile, 'w', **profile) as dst:
            dst.write(arr_diff, 1)
        


