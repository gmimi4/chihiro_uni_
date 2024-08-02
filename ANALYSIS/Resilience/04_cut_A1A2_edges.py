# -*- coding: utf-8 -*-
"""
# A1 right edge and A2 left edge has error pixels, so exclude
"""

import os
import glob
import rasterio
import numpy as np
from tqdm import tqdm
from rasterio.windows import Window

# in_dir = r"D:\Malaysia\02_Timeseries\Resilience\03_change_point"
in_dir = r"D:\Malaysia\02_Timeseries\Resilience\03_halfperiod"
out_dir = in_dir

tifs_list = glob.glob(in_dir + os.sep + "*.tif")
tifs_list_a1a2 = [t for t in tifs_list if "A1" in os.path.basename(t) or "A2" in os.path.basename(t)]

""" # cut A1 A2 edges"""
# how many cut?
elim_pixelnum = 2

for t in tifs_list_a1a2:
    filename = os.path.basename(t)[:-4]
    src = rasterio.open(t) #do not close src
    arr = src.read(1)
    rows,cols = arr.shape
    meta = src.meta
    
    if "A1" in filename: #cut 2 right columns #Window(col_off, row_off, width, height)
        crop_window = Window(0, 0, cols - elim_pixelnum, rows)        
    else: #cut 2 left columns
        crop_window = Window(elim_pixelnum, 0, cols - elim_pixelnum, rows)  
        # crop raster
        
    cropped_data = src.read(1, window=crop_window)
    
    # Update the transform to reflect the new window
    new_transform = src.window_transform(crop_window)
    
    #update
    meta.update({"transform":new_transform,"width":cropped_data.shape[1],"height":cropped_data.shape[0]})
      
    outfile = out_dir + os.sep + f"{filename}cut.tif"
    with rasterio.open(outfile, 'w', **meta) as dst:
        dst.write(cropped_data, 1)
                
    
    


        
       
    
        





