# -*- coding: utf-8 -*-
"""
# change to cv
# 
"""

import numpy as np
import pandas as pd
import os,sys
import matplotlib.pyplot as plt
import glob
import rasterio
from tqdm import tqdm
# from statsmodels.tsa.seasonal import seasonal_decompose
from rasterio.plot import show
from rasterio.crs import CRS
with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
    rio_crs = CRS.from_epsg(4326)
    proj_crs = CRS.from_user_input(rio_crs)

PageName = 'A1'
csv_dir = rf"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels\{PageName}"
# csv_dir = f'/Volumes/PortableSSD/MAlaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels/{PageName}'
p_val_tif = rf'D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_01\{PageName}_p_values_importance_2013-2022.tif'
# p_val_tif = f'/Volumes/SSD_2/Malaysia/02_Timeseries/CPA_CPR/2_out_ras/p_01/{PageName}_p_values_importance_2013-2022.tif'
importance_tif_dir = os.path.dirname(p_val_tif)
out_dir = r'D:\Malaysia\02_Timeseries\Sensitivity\1_std\std_ras\p_01'
# out_dir = '/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/1_std/std_ras/p_01'
# coef_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\1_quadratic\p_010"

# Palmのあるインデックス取得
# palm_idx_df = pd.read_csv(palm_indx_txt, header=None)
# palm_idx_list = palm_idx_df.values.tolist()
# palm_idx_list = [t[0] for t in palm_idx_list]

# # p_valが有意なインデックス取得
# p_src = rasterio.open(p_val_tif)
# p_arr = p_src.read(1)
# p_arr_1d = np.ravel(p_arr)
# p_idx = list(np.where(p_arr_1d < 0.1)[0]) #0.05
# p_idx = [13330]

#両方のインデックスの積集合
# use_idx = list(set(palm_idx_list) & set(p_idx))

csvs = glob.glob(os.path.join(csv_dir,"*.csv"))
# csvs_use = [c for c in csvs if int(os.path.basename(c)[:-4]) in use_idx]
csvs_use = csvs

startyear = 2002
endyear = 2012


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
    
    """ #detrend for CV *zscoring generate 0 mean""" 
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
    
    
    """ #cal CV"""
    ## use original detrended data
     
    cv_all_var = {} #for each month
    cv_overall_var = {} #for overall sum
    for variable in  vars_list:     
        df_var = df_csv_z.loc[:,variable] #use original data
        
        cv_month_var ={}
        for m in months:
            specific_month_rows = df_var[df_var.index.month == m]
            monthly_mean = specific_month_rows.mean(skipna=True)
            monthly_std = specific_month_rows.std(skipna=True)
            var_cv = monthly_std/monthly_mean

            cv_month_var[m] = var_cv
        
        ## sum cv for a whole year
        cv_sum_var = sum(cv_month_var[x] for x in cv_month_var)
        
        cv_all_var[variable] = cv_month_var #cv each month
        cv_overall_var[variable] = cv_sum_var #overall cv val <- use this
    
    ## sample plot (later)
        

    cv_files[int(filename)] = cv_overall_var

        


""" # Export tif for cv of each variale """ 
with rasterio.open(p_val_tif) as src: # pval tif as sample tif
    arr = src.read(1)
    profile=src.profile
    height, width = arr.shape[0],arr.shape[1]
    
#念のためsort by filename (idx name)
all_resid_sort = sorted(cv_files.items())

for variable in  vars_list: 
    var_cv_arr = np.array([c[1][variable] for c in all_resid_sort]) #順番通りに取り出しているはず 
    var_cv_reshape = var_cv_arr.reshape((height, width)) 

    outfile = os.path.join(out_dir,"{PageName}_cv_{variable}.tif")
    with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
        with rasterio.open(outfile, "w", **profile) as dst:
            dst.write(var_cv_reshape, 1)









