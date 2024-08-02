# -*- coding: utf-8 -*-
"""
# obrain residuals from STL
# make it as overall residuals for each variable
"""

import numpy as np
import pandas as pd
import os,sys
import matplotlib.pyplot as plt
import glob
import rasterio
from tqdm import tqdm
import copy
import warnings
from rasterio.plot import show
from rasterio.crs import CRS
with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
    rio_crs = CRS.from_epsg(4326)
    proj_crs = CRS.from_user_input(rio_crs)

# in_dir_cv = r"D:\Malaysia\02_Timeseries\Sensitivity\1_cv\1_cv_ras\p_01"
in_dir_cv = '/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/1_cv/1_cv_ras/p_01'

in_dir_importane = '/Volumes/SSD_2/Malaysia/02_Timeseries/CPA_CPR/2_out_ras/p_01' #no time effects
# in_dir_importane = '/Volumes/SSD_2/Malaysia/02_Timeseries/CPA_CPR/2_out_ras/timeeffects' #with time effect

out_dir = '/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/1_cv/2_cv_sensitivity_allperiod' #no time effects
# out_dir = '/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/1_cv/2_cv_sensitivity_allperiod/timeeffect' #with time effect

PageName_list = ["A1","A2","A3","A4"]

# period_str = "2002-2022"
period_str = sys.argv[1]

for page in PageName_list:
    """ # process by PageName"""
    # page ="A1"
    
    """ # cv and ratio to GPP"""
    csv_use = glob.glob(os.path.join(in_dir_cv,f"*{page}*{period_str}*.csv"))[0]
    # csvs_use = [c for c in csvs if int(os.path.basename(c)[:-4]) in use_idx]
    
    df_cv = pd.read_csv(csv_use)
    df_cv = df_cv.drop(df_cv.columns[[0]], axis=1)
    
    # make list of var excepting "GOSIF"
    var_list = df_cv.columns.to_list()
    var_list.remove("GOSIF")
    
    ## calc ratio: GOSIF to variable
    for var in var_list:
        df_cv[var+"r"] = df_cv["GOSIF"] / df_cv[var]
        
        
    """ # calc weights for variable in a pixel"""
    
    """ # p_valが有意なインデックス取得 (** at present, no p filtering)"""
    p_val_tif = glob.glob(in_dir_importane +os.sep+f"{page}*_p_*{period_str}*.tif")[0]
    # periods_list =[os.path.basename(p)[:-4].split("_")[-1] for p in p_val_tifs]
    
    with rasterio.open(p_val_tif) as p_src:
        p_arr = p_src.read(1)
        p_arr_1d = np.ravel(p_arr)
        p_idx = list(np.where(p_arr_1d < 0.1)[0]) #0.05
        
    ### but there are a few pixel left if filtering by p...
    
    
    """ # pick importances"""
    importance_tifs = glob.glob(in_dir_importane +os.sep+f"*{page}*_{period_str}*.tif")
    
    importance_var_arr_dic = {}
    for var in var_list:
        tif = [t for t in importance_tifs if var in t][0]
        with rasterio.open(tif) as src:
            arr = src.read(1)
            arr_flat = arr.ravel()
            importance_var_arr_dic[var] = arr_flat
     
        
    ## this is goal dic        
    importance_pixel_dic = {} #ピクセルごとのcoefをvarの辞書形式で回収する.
    for i in tqdm(range(len(arr_flat))):
        importance_pixel={}
        for var in var_list:
            varcoef = importance_var_arr_dic[var][i]
            importance_pixel[var]=varcoef
        importance_pixel_dic[i] = importance_pixel
    
    
    """ # calc weight"""
    ## if index is not significant (p>0.1), same weight applied to all vars
    importance_weights = copy.deepcopy(importance_pixel_dic)
    
    for i, coefdic in tqdm(importance_weights.items()):
        # coefdic = importance_weights[10930] #not significant
        # coefdic = importance_weights[11044] #significant
        
        coeflist = [abs(c) for var, c in coefdic.items()] #絶対値
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try: #coeflistが全部nanだとWarningが出る
                sumcoef = np.nansum(coeflist)
                
                coefdic_wei ={}
                for variable in var_list:
                    coef = abs(coefdic[variable])
                    coef_wei = coef / sumcoef
                    coefdic_wei[variable] =coef_wei
            except ValueError as e:
                for variable in var_list:
                    coefdic_wei[variable] =np.nan
        
        importance_weights[i] = coefdic_wei
        #check
        # np.nansum([s for s in importance_weights[i].values()])
        
        
    """ # multiply cv ratio by weight, and sum"""
    sensitivity = {}
    for i, ratios in df_cv.iterrows():
        # i =11044 #significant #10930 #not significant
        # ratios = df_cv.loc[i,:]
        sensitivity_pixel = []
        for var in var_list:
            ratio_var = ratios[f"{var}r"] #ratios is series
            wei_var = importance_weights[i][var]
            sensitivity_var = ratio_var * wei_var
            
            sensitivity_pixel.append(sensitivity_var)
        
        if np.isnan(sensitivity_pixel).all(): #all variable cv*weight are nan
            sensitivity_pixel_sum = np.nan
        else:
            sensitivity_pixel_sum = np.nansum(sensitivity_pixel)
        
        sensitivity[i] = sensitivity_pixel_sum
            
        
    """　# tifのExport """
    #念のためsort
    all_sensitivity_sort = sorted(sensitivity.items())
    all_sensitivity_arr = np.array([t[1] for t in all_sensitivity_sort]) #順番通りに取り出しているはず 
    
    
    with rasterio.open(importance_tifs[0]) as src:
        arr = src.read(1)
        profile=src.profile
        height, width = arr.shape[0],arr.shape[1]
    
    
    all_sensitivity_reshape = all_sensitivity_arr.reshape((height, width)) 
    
    outfile = os.path.join(out_dir,f"{page}_sensitivity_cv_allperiod_{period_str}.tif")
    with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
        with rasterio.open(outfile, "w", **profile) as dst:
            dst.write(all_sensitivity_reshape, 1)

#plot
# ras = rasterio.open(outfile)
# show(ras)




