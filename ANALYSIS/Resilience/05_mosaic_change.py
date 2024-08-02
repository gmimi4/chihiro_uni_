# -*- coding: utf-8 -*-
"""
# A1 right edge and A2 left edge has error pixels, so exclude
"""

import os
import glob
import rasterio
from rasterio.merge import merge
import numpy as np
from tqdm import tqdm

# in_dir = r"D:\Malaysia\02_Timeseries\Resilience\03_change_point"
in_dir = r"D:\Malaysia\02_Timeseries\Resilience\03_halfperiod"
out_dir = in_dir + os.sep + "_mosaic"

tifs_list = glob.glob(in_dir + os.sep + "*.tif")

tifs_list_cut = []
for t in tifs_list:
    filename = os.path.basename(t)
    if "A1" in filename or "A2" in filename:
        if "cut.tif" in filename:
            tifs_list_cut.append(t)
        else:
            pass
   
    else:
       tifs_list_cut.append(t)
    

variable_list = [os.path.basename(t)[:-4].split("_")[1] for t in tifs_list_cut]
variable_list = list(set(variable_list))

for variable in variable_list: 
    
    tifs_p = [t for t in tifs_list_cut if variable in t]
            
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
        
    
    


        
       
    
        





