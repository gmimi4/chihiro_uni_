# -*- coding: utf-8 -*-
"""
# select pixels changed its sign to water stressed response

"""

import os
import glob
import rasterio
import numpy as np
import pandas as pd

in_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_01\_mosaic"
out_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\4_stressed_pixels"

variable_list = [
                "rain", 
                "temp",
                "VPD",
                "Et",
                "Eb",
                "SM",
                "VOD"
                 ]

def raster_array(tif_path): #2d
    with rasterio.open(tif_path) as src_def:
        arr_def = src_def.read(1)  
    return arr_def


tifs = glob.glob(in_dir + os.sep + "*.tif")
tifs = [t for t in tifs if (not "p_values" in t) and (not "r_square" in t) ]

# periods = list(set([os.path.basename(t)[:-4].split("_")[-1] for t in tifs]))
# periods = ['2002-2012', '2013-2022']

""" # sample tif"""
with rasterio.open(tifs[0]) as src:
    meta = src.meta
    arr_sample = src.read(1)
    height, width = arr_sample.shape[0],arr_sample.shape[1]
    # meta.update({'dtype':'uint8','nodata':0 })
        
    
## make flat array by var
out_tifs =[]
for var in variable_list:
    tifs_var = [t for t in tifs if var in t]
    tif_early =[t for t in tifs_var if "2002-2012" in t][0]
    tif_later =[t for t in tifs_var if "2013-2022" in t][0]
    
    arr_early = raster_array(tif_early)
    arr_later = raster_array(tif_later)
    
    """ # find pixels that have become water-stressed"""
    if (var == "VOD") or (var == "rain") or (var == "SM"):
        # row, col
        arr_early_ok = np.where(arr_early <=0) # no stress with neative or neutral relation 
        arr_later_stressed = np.where(arr_later >0) # stressed with positive replation
    
    else: #Temp, Et, Eb, VPD, which are not wanted when stressed
        arr_early_ok = np.where(arr_early >=0) # no stress with neative or neutral relation 
        arr_later_stressed = np.where(arr_later <0) # stressed with positive replation
    
    arr_early_ok_list = [(arr_early_ok[0][i],arr_early_ok[1][i]) for i in range(len(arr_early_ok[0]))]
    arr_later_stressed_list = [(arr_later_stressed[0][i],arr_later_stressed[1][i]) for i in range(len(arr_later_stressed[0]))]
    
    
    ## find common pixels
    index_vulnerable = list(set(arr_early_ok_list)&set(arr_later_stressed_list))
    ## arrange index form
    # https://stackoverflow.com/questions/69257664/how-to-replace-values-of-a-2d-array-by-an-array-of-indices-in-python
    ind = tuple(np.array(index_vulnerable).T) #back to like rows and cols form
    
    """ # get difference in abs"""
    zero_div_values = np.zeros(arr_early.shape) 
    zero_div_values = np.where(zero_div_values==0, np.nan, np.nan) #計算できなかったときnanを入れる
    
    if (var == "VOD") or (var == "rain") or (var == "SM"):
        # later positive - early minus or zero
        arr_diff = arr_later - arr_early
        # arr_ratio = np.divide(arr_diff, arr_later, out=zero_div_values, where=(arr_later!=0))
    else:
        # early plus or zero - later negative 
        arr_diff = arr_early - arr_later
        # arr_ratio = arr_ratio = np.divide(arr_diff, arr_early, out=zero_div_values, where=(arr_early!=0))
    
    """ # put diff value in pixels where found as vulnerable avobe"""
    arr_vulnerable = np.zeros((height, width)) # 0 array
    arr_vulnerable[:,:] = np.nan #nan array
    arr_vulnerable[ind] = arr_diff[ind]
    # arr_vulnerable[ind] = arr_ratio[ind]
    

    outfile = out_dir + os.sep + f"sign_change_{var}.tif"
    
    with rasterio.open(outfile, 'w', **meta) as dst:
        dst.write(arr_vulnerable, 1)
        
    out_tifs.append(outfile)
    
    
    
""" # Extract pixels where all vars had changes in signs"""
## try several number of vars which have made change in sign

## ピクセル数を割合で出す
sample_tif = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_01\_mosaic\mosaic_p_values_importance_2002-2022.tif"
arr_sample = np.ravel(raster_array(sample_tif))
arr_sample_valid = [a for a in arr_sample if a>=0]
num_all_pixel = len(arr_sample_valid)

ratio={}

for valid_num in [1,2,3,4,5,6,7]:
    
    arr_1d_list = [np.ravel(raster_array(t)) for t in out_tifs]
     
    arr_stacks = np.stack(arr_1d_list)
    # NG: arr_stacks_valid = np.sum(arr_stacks, axis=0) # almost all pixels gone
    
    # Count non-NaN values along the first axis
    valid_count = np.sum(~np.isnan(arr_stacks), axis=0)
    # Mask for valid sums (where count > more than half)
    valid_mask = valid_count >= valid_num
    # set nan array
    result = np.full(arr_stacks.shape[1:], np.nan)
    # Calculate the sum along the first axis where the mask is true
    result[valid_mask] = np.nansum(arr_stacks, axis=0)[valid_mask]
    
    ## reshape and export
    arr_reshape = np.reshape(result, [height, width])
           
    outfile2 = out_dir + os.sep + f"sign_change_morethan{valid_num}.tif"
    
    with rasterio.open(outfile2, 'w', **meta) as dst:
        dst.write(arr_reshape, 1) 
    
    
    ### count pixel number
    num_valid_mask = [v for v in valid_mask if v==True]
    num_ratio = len(num_valid_mask)/num_all_pixel
    
    ratio[valid_num] = num_ratio
    

df_ratio = pd.DataFrame(ratio.values(), index=ratio.keys())
df_ratio.to_csv(out_dir+os.sep+"num_ratio.txt")
    
    
    
    
    
