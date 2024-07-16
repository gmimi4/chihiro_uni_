# -*- coding: utf-8 -*-
"""
# select the largest importance variable
# grouping

"""
import os
import glob
import rasterio
import numpy as np
from tqdm import tqdm

in_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_01\_mosaic"
out_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\3_major_variable"

variable_dic = {
                   # "GOSIF",
                "rain":1, 
                "temp":2,
                "VPD":2,
                "Et":4,
                "Eb":5,
                "SM":5,
                "VOD":4,
                 }


""" # select largest importance val from stacked ras"""
tifs = glob.glob(in_dir + os.sep + "*.tif")

periods = list(set([os.path.basename(t)[:-4].split("_")[-1] for t in tifs]))

""" # sample tif"""
with rasterio.open(tifs[0]) as src:
    meta = src.meta
    arr_sample = src.read(1)
    height, width = arr_sample.shape[0],arr_sample.shape[1] 
    
    

for peri in periods:
    tifs_peri = [t for t in tifs if (peri in t) and (not "p_values" in t) ]
    
    ## make flat array by var
    var_val_dic ={}
    for var in variable_dic.keys():
        tif = [t for t in tifs_peri if var in t][0]
        with rasterio.open(tif) as src:
            arr = src.read(1)
            arr_flat = np.ravel(arr)
            var_val_dic[var] = arr_flat
        

    total_num = len(var_val_dic["rain"]) # want to know num of pixels
    
    ### obtain values in index i for each var  
    var_val_major = {}
    for i in tqdm(range(total_num)):
        var_val_dic_i = {}
        for var in variable_dic.keys():
            var_val_dic_i[var] = abs(var_val_dic[var][i])
        
        ### obtain var key with the largest imoprtance (abs)
        # not allowed <- allowing nan because nan will mean no significanse
        if not np.any(np.isnan(list(var_val_dic_i.values()))) and not np.sum(list(var_val_dic_i.values())) ==0:
            # record the largest var in i th data in a numeric val
            major_var = max(var_val_dic_i, key=var_val_dic_i.get)
            var_val_major[i] = variable_dic[major_var]
        # if nan exist in any variable, 0 is input
        else:
            var_val_major[i] = 0
    
    ### put to flat array
    #念のためindx順にソート
    ras_dic_sort = sorted(var_val_major.items())
    #　
    major_arr = np.full(len(ras_dic_sort), np.nan)    
    for i in ras_dic_sort:
        arri = i[0]
        arrval = i[1]
        np.put(major_arr, [arri], arrval)
        
    # reshape
    major_arr_re = major_arr.reshape((height, width))


    outfile = out_dir + os.sep + f"major_{peri}.tif"
    meta.update({'dtype':'uint8','nodata':0 })
    with rasterio.open(outfile, 'w', **meta) as dst:
        dst.write(major_arr_re, 1)
        
    
    


        
       
    
        





