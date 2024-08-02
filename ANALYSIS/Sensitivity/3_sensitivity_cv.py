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

PageName = 'A1'
period_str = "2002-2022"
csv_dir = rf"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels\{PageName}"
# csv_dir = f'/Volumes/PortableSSD/MAlaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels/{PageName}'
in_dir_importane = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_01"
p_val_tif = in_dir_importane + os.sep + f"{PageName}_p_values_importance_{period_str}.tif"
# p_val_tif = f'/Volumes/SSD_2/Malaysia/02_Timeseries/CPA_CPR/2_out_ras/p_01/{PageName}_p_values_importance_2013-2022.tif' #as sample tif
out_dir = r'D:\Malaysia\02_Timeseries\Sensitivity\1_std\std_ras\p_01'
# out_dir = f'/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/1_std/std_ras/p_01/{PageName}'
os.makedirs(out_dir,exist_ok=True)
# coef_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\1_quadratic\p_010"

csvs = glob.glob(os.path.join(csv_dir,"*.csv"))
# csvs_use = [c for c in csvs if int(os.path.basename(c)[:-4]) in use_idx]
csvs_use = csvs

startyear = 2002
endyear = 2022


""" # dfからMonthly mean dicを得る"""
months = [m for m in range(1,13)]
def get_monthly_mean_dic(df):
    #df=df_seasonal
    month_dic={}
    for m in months:
        month_row = df[df.index.month == m]
        monthly_mean = month_row.mean()
        month_dic[m] = monthly_mean
    
    return month_dic


""" # detrend data, and obtain cv for each month, then sum up"""
cv_files = {}
for csvfile in tqdm(csvs_use):    
    # csvfile = [c for c in csvs_use if "15706" in c][0] #4447 #8682
    # csvfile = r"D:\Malaysia\02_Timeseries\CPA_CPR\1_vars_at_pixels\1838.csv"
    filename = os.path.basename(csvfile)[:-4]
    df_csv = pd.read_csv(csvfile, index_col = 'datetime',
                         parse_dates=['datetime'])

    """ #AMSREとAMSRを補正する"""
    pi_e = df_csv.loc[:,["SMDSCE","VODDSCE"]]
    pi_2 = df_csv.loc[:,["SMDSC2","VODDSC2"]]
    # calc median
    pi_e_med = pi_e.median()
    pi_2_med = pi_2.median()
    ratio_sm = pi_2_med.SMDSC2 /pi_e_med.SMDSCE
    ratio_vod = pi_2_med.VODDSC2 /pi_e_med.VODDSCE
    
    df_csv["SMDSCErev"] = df_csv["SMDSCE"] *ratio_sm
    df_csv["VODDSCErev"] = df_csv["VODDSCE"] *ratio_vod
    
    
    """ #SMとVODは平均にする"""
    #skipnaでいけた
    df_csv["SM"] = df_csv[["SMDSCE", "SMDSC2"]].mean(skipna=True, axis='columns')
    df_csv["VOD"] = df_csv[["VODDSCE", "VODDSC2"]].mean(skipna=True, axis='columns')
    df_csv= df_csv.drop(["SMDSCE","SMDSC2","VODDSCE","VODDSC2","SMDSCErev", "VODDSCErev"], axis=1)
    
    
    vars_list = df_csv.columns.tolist()
    
    
    """ # slice period """
    df_csv = df_csv[((df_csv.index.year >= startyear) & (df_csv.index.year <= endyear)) ]
    
    
    """ #detrend for CV *zscoring generates 0 mean""" 
    # obtain monthly mean for each month
    monthly_mean_org = get_monthly_mean_dic(df_csv)
    
    df_csv_de = df_csv.copy()
    
    for var in vars_list:
        df_csv_de[f"{var}de"] = np.nan
        df_var = df_csv_de.loc[:,var]
        
        for m in months:
            specific_month_rows = df_var[df_var.index.month == m]
            monthly_mean = monthly_mean_org[m][var]
            
            ##インデックスで抽出して元のデータフレームの新規列にzscoreを入れる
            specific_month_idx = specific_month_rows.index.tolist()
            df_csv_de.loc[specific_month_idx, f"{var}de"] = df_csv_de[var]-monthly_mean

    
    
    """ #(pixel単位で)月平均とstdで zscoring""" 
    # obtain monthly mean for each month
    # monthly_mean_dic = monthly_mean_dic(df_csv)
    
    df_csv_z = df_csv_de.copy()
    
    # monthly_mean_std_dic = {}
    for var in vars_list:
        df_csv_z[f"{var}z"] = np.nan
        df_var = df_csv_z.loc[:,var]
        month_dic = {}
        for m in months:
            specific_month_rows = df_var[df_var.index.month == m]
            monthly_mean = specific_month_rows.mean(skipna=True)
            monthly_std = specific_month_rows.std(skipna=True)
            month_dic[m] = [monthly_mean, monthly_std]
            
            ##インデックスで抽出して元のデータフレームの新規列にzscoreを入れる
            specific_month_idx = specific_month_rows.index.tolist()
            df_csv_z.loc[specific_month_idx, f"{var}z"] = (df_csv_z[var]-monthly_mean)/monthly_std
    
        # monthly_mean_std_dic[var] = month_dic #確認用
    
        
    """ #(pixel単位で)Standarized (0-1)""" 
    
    df_csv_s = df_csv_z.copy()
    
    # monthly_mean_std_dic = {}
    for var in vars_list:
        df_csv_s[f"{var}s"] = np.nan
        df_var = df_csv_s.loc[:,var]
        for m in months:
            specific_month_rows = df_var[df_var.index.month == m]
            monthly_mean = specific_month_rows.mean(skipna=True)
            monthly_min = specific_month_rows.min(skipna=True) #min
            monthly_max = specific_month_rows.max(skipna=True) #max
    
            ##インデックスで抽出して元のデータフレームの新規列にsrandarizedを入れる
            specific_month_idx = specific_month_rows.index.tolist()
            df_csv_s.loc[specific_month_idx, f"{var}s"] = (df_csv_z[var]-monthly_min)/(monthly_max - monthly_min)
    
        # monthly_mean_std_dic[var] = month_dic #確認用
    
    
    """ #cal CV"""    
    cv_all_var = {} #for each month
    for variable in  vars_list:     
        df_var = df_csv_s.loc[:,variable+"s"]
        
        cv_month_var ={}
        for m in months:
            specific_month_rows = df_var[df_var.index.month == m]
            monthly_mean = specific_month_rows.mean(skipna=True)
            monthly_std = specific_month_rows.std(skipna=True)
            var_cv = monthly_std/monthly_mean

            cv_month_var[m] = var_cv
        
        ## sum cv for a whole year
        # cv_sum_var = sum(cv_month_var[x] for x in cv_month_var)
        
        cv_all_var[variable] = cv_month_var #cv each month
        
        
    """ # calc ratio for each month """
    vars_list_remove = vars_list.copy()
    vars_list_remove.remove("GOSIF")
    
    ratio_month_var ={} #ratio for each var <- use this
    
    for variable in vars_list_remove:
        raito_month = [] # ratio for each month
        for m in months:
            ratio_var =  cv_all_var["GOSIF"][m] / cv_all_var[variable][m]
            raito_month.append(ratio_var)
            
        ratio_mean = np.mean(raito_month)
        ratio_month_var[variable] = ratio_mean
        
    
    cv_files[int(filename)] = ratio_month_var

    

""" # calc weights for variable in a pixel"""

""" # p_valが有意なインデックス取得 (** at present, no p filtering)"""
### there are a few pixel left if filtering by p...
with rasterio.open(p_val_tif) as p_src:
    p_arr = p_src.read(1)
    p_arr_1d = np.ravel(p_arr)
    p_idx = list(np.where(p_arr_1d < 0.1)[0]) #0.05
    

""" # pick importances"""
importance_tifs = glob.glob(in_dir_importane +os.sep+f"*{PageName}*_{period_str}*.tif")

importance_var_arr_dic = {}
for var in vars_list_remove:
    tif = [t for t in importance_tifs if var in t][0]
    with rasterio.open(tif) as src:
        arr = src.read(1)
        arr_flat = arr.ravel()
        importance_var_arr_dic[var] = arr_flat
 
    
## this is goal dic        
importance_pixel_dic = {} #ピクセルごとのcoefをvarの辞書形式で回収する.
for i in tqdm(range(len(arr_flat))):
    importance_pixel={}
    for var in vars_list_remove:
        varcoef = importance_var_arr_dic[var][i]
        importance_pixel[var]=varcoef
    importance_pixel_dic[i] = importance_pixel


""" # calc weight"""
## if index is not significant (p>0.1), same weight applied to all vars
importance_weights = copy.deepcopy(importance_pixel_dic)
import warnings

for i, coefdic in tqdm(importance_weights.items()):
    # coefdic = importance_weights[10930] #not significant
    # coefdic = importance_weights[11044] #significant
    
    coeflist = [abs(c) for var, c in coefdic.items()] #絶対値
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try: #coeflistが全部nanだとWarningが出る
            sumcoef = np.nansum(coeflist)
            
            coefdic_wei ={}
            for variable in vars_list_remove:
                coef = abs(coefdic[variable])
                coef_wei = coef / sumcoef
                coefdic_wei[variable] =coef_wei
        except ValueError as e:
            for variable in vars_list_remove:
                coefdic_wei[variable] =np.nan
    
    importance_weights[i] = coefdic_wei
    #check
    # np.nansum([s for s in importance_weights[i].values()])
    
    
""" # multiply ratio by weight, and sum"""
sensitivity = {}
for i, ratios in cv_files.items():
    # i =11044 #significant #10930 #not significant
    # ratios = df_cv.loc[i,:]
    sensitivity_pixel = []
    for var in vars_list_remove:
        ratio_var = ratios[f"{var}"]
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

outfile = os.path.join(out_dir,f"{PageName}_sensitivity_cv_{period_str}.tif")
with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
    with rasterio.open(outfile, "w", **profile) as dst:
        dst.write(all_sensitivity_reshape, 1)








