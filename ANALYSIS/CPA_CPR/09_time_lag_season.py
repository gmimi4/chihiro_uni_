# -*- coding: utf-8 -*-
"""
# see when variable correlates with SIF using pearson coef 
#Affine: https://www.perrygeo.com/python-affine-transforms.html
#no detrending
#GOSIF
"""

import numpy as np
from scipy import stats
import pandas as pd
import os,sys
import glob
from tqdm import tqdm
import rasterio

startyear = 2000
endyear = 2023

# time_lag = 1 #
time_lag = sys.argv[1]
    
for page in ["A1","A2","A3","A4"]:
    # page = "A3"
    # in_dir = rf"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels\{page}"
    in_dir = f'/Volumes/PortableSSD/Malaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels_until2023/{page}'
    csv_file_list = glob.glob(in_dir + os.sep + "*.csv")
    # out_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\6_time_lag" + os.sep + f"lag_{str(time_lag)}"
    out_dir = "/Volumes/SSD_2/Malaysia/02_Timeseries/CPA_CPR/6_time_lag/GOSIF/_season" + os.sep + f"lag_{str(time_lag)}"
    os.makedirs(out_dir, exist_ok=True)
    
    ### time series csvをつくったラスターのどれか
    # sample_tif = r"F:\MAlaysia\SIF\GOSIF\02_tif_age_adjusted\res_01\extent\GOSIF_2000081_extent_adj_res01_extent_A1.tif"
    sample_tif = f'/Volumes/PortableSSD/MAlaysia/SIF/GOSIF/02_tif_age_adjusted/res_01/extent/GOSIF_2000081_extent_adj_res01_extent_{page}.tif'
    with rasterio.open(sample_tif) as src:
        src_arr = src.read(1)
        meta = src.meta
        transform = src.transform
        height, width = src_arr.shape[0], src_arr.shape[1]
        # profile = src.profile
        
    ## 参照しているラスターのAffineのheight pixelはマイナスになっていてほしい
    meta.update({"nodata":np.nan})
    
    
    use_vars = ["rain","temp","VPD","Et","Eb","SM","VOD"]
    
    """### person correlationをピクセルごとに集計する """
    idx_pearson_dic = {} #pearson and pval for each var at all pixels
    for csvfile in tqdm(csv_file_list):
        # csvfile = in_dir + os.sep + "15706.csv"
        
        idx = os.path.basename(csvfile)[:-4]
        df_csv = pd.read_csv(csvfile, index_col = 'datetime',
                             parse_dates=['datetime'])
        
    
        """ # AMSREとAMSR2のギャップを補正する"""    
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
        # try: #AMSREとAMSR2を一列にする
        df_csv["SM"] = df_csv[["SMDSCErev", "SMDSC2"]].mean(skipna=True, axis='columns')
        df_csv["VOD"] = df_csv[["VODDSCErev", "VODDSC2"]].mean(skipna=True, axis='columns')
        df_csv= df_csv.drop(["SMDSCE","SMDSC2","VODDSCE","VODDSC2", "SMDSCErev", "VODDSCErev"], axis=1)   
        
        """ # set period"""
        df_csv = df_csv[((df_csv.index.year >= startyear) & (df_csv.index.year <= endyear)) ]
        
        """ #(pixel単位で)月平均とstdで zscoring""" 
        months = [m+1 for m in range(12)]
        vars_list = df_csv.columns.tolist()
        
        df_csv_z = df_csv.copy()
        
        ### obtain monthly mean and std as dict
        # if a certain month has only one data, std is nan and z score can not be calculated
        # monthly_mean_std_dic = {}
        # for var in vars_list:
        #     df_csv_z[f"{var}z"] = np.nan
        #     df_var = df_csv_z.loc[:,var]
        #     month_dic = {}
        #     for m in months:
        #         specific_month_rows = df_var[df_var.index.month == m]
        #         monthly_mean = specific_month_rows.mean(skipna=True)
        #         monthly_std = specific_month_rows.std(skipna=True)
        #         month_dic[m] = [monthly_mean, monthly_std]
                
        #         ##インデックスで抽出して元のデータフレームの新規列にzscoreを入れる
        #         specific_month_idx = specific_month_rows.index.tolist()
        #         df_csv_z.loc[specific_month_idx, f"{var}z"] = (df_csv_z[var]-monthly_mean)/monthly_std
        #         # df_csv_z.loc[specific_month_idx, f"{var}z"] = df_csv_z[var]-monthly_mean
        
        #     monthly_mean_std_dic[var] = month_dic #確認用
        
        """ only z scoring"""
        for var in vars_list:
            df_csv_z[f"{var}z"] = np.nan
            df_var = df_csv_z.loc[:,var]                
            ## z scoring
            var_mean = df_csv_z.loc[:,f"{var}"].mean(skipna=True)
            var_std = df_csv_z.loc[:,f"{var}"].std(skipna=True)
            df_csv_z.loc[:, f"{var}z"] = (df_csv_z[f"{var}"]-var_mean)/var_std
            
        ### nan削除
        df_valid = df_csv_z.dropna() #nanがあれば除外
        df_valid = df_valid.drop(vars_list, axis=1) # use z-scored data
        
        if len(df_valid) ==0:
            r_p_dic = {v:[] for v in use_vars} #pearson and pval for each var are np.nan
            idx_pearson_dic[idx] = r_p_dic
        
        else:    
            """　# shift Y:GOSIF by time lag """     
            # reset datetime index
            df_valid_reset = df_valid.reset_index()
            # extract gosif
            vars_listz = [v + "z" for v in vars_list if v != "GOSIF"]
            df_valid_sifonly = df_valid_reset.drop(vars_listz + ["datetime"], axis=1)
            # extract vars z scored only
            df_valid_varonly = df_valid_reset.drop(["GOSIFz","datetime"], axis=1)
            
            ### shift gosif data set by lag
            empty_data = [np.nan for _ in range(int(time_lag))] # [np.nan, np.nan, ...]
            df_empty = pd.DataFrame({"GOSIFz":empty_data})
            
            # concat empty df and laged sif df
            df_valid_sifonly_lag = pd.concat([df_empty, df_valid_sifonly], axis=0, ignore_index=True)
        
            # concat laged sif df and vars df
            df_valid_lag = pd.concat([df_valid_sifonly_lag, df_valid_varonly], axis=1)
            
            # remove nan row
            df_valid_lag = df_valid_lag.dropna(how="any")
            
        
            """　#Select Dataframe """    
            Y_norm = df_valid_lag['GOSIFz'] #monthly z scoringしたものを使う
            X_norm = df_valid_lag.drop(['GOSIFz'], axis=1)
            
            
            """　# Process if valid length of data is more than 6 """  
            if not len(Y_norm) >5:
                r_p_dic = {v:[] for v in use_vars} 
                idx_pearson_dic[idx] = r_p_dic
            
            else:
            
                varzs = [col for col in X_norm.columns]
                
                r_p_dic = {} #pearson and pval for each var
                for var in use_vars: #with "z" in name
                    X_norm_var = X_norm.loc[:,var+"z"]
                    r, pval = stats.pearsonr(X_norm_var, Y_norm)
                    
                    r_p_dic[var] = [r, pval]
                 
                    
                # pur pearson and pval of each var and at each pixel to all dic
                idx_pearson_dic[idx] = r_p_dic
            
    
    """### 変数ごとにラスターに変換 """
    
    for vari in use_vars:
        # vari = "rain"
        r_dic, pval_dic = {},{}
        for i,r_pval in idx_pearson_dic.items():
            # i=198
            # r_pval = idx_pearson_dic[str(i)]
            r_pval_var = r_pval[vari]
            if len(r_pval_var)>0:
                r_var = r_pval_var[0]
                p_var = r_pval_var[1]
                
                r_dic[int(i)] = r_var
                pval_dic[int(i)] = p_var
            else:
                r_dic[int(i)] = np.nan
                pval_dic[int(i)] = np.nan  
    
        
        for k, dicval in {"pearson":r_dic, "pval":pval_dic}.items():
            #念のためindx順にソート
            ras_dic_sort = sorted(dicval.items()) #タプルになった(indx, importance)
            
            #これに入れる
            importance_arr = np.full(len(ras_dic_sort), np.nan)
            
            for i in ras_dic_sort:
                arri = i[0]
                arrval = i[1]
                np.put(importance_arr, [arri], arrval)
                
            
            # reshape
            importance_arr_re = importance_arr.reshape((height, width))
            
            out_file = out_dir +os.sep +f"{page}_{vari}_{k}_lag{time_lag}_{str(startyear)}-{str(endyear)}.tif"
            with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
                with rasterio.open(out_file, 'w', **meta) as dst:
                  dst.write(importance_arr_re, 1)


