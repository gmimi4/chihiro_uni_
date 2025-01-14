# -*- coding: utf-8 -*-
"""
PCA用にピクセルごとに目的変数と変数を時系列に並べたcsvを出力する。
陸域のrows*colmnsの数だけ出力される
@author: chihiro

"""

import os, sys
import glob
# import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
import time
import datetime
from tqdm import tqdm

# csv_parent_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\0_vars_timeseries\EVI"
csv_parent_dir = '/Volumes/SSD_2/Malaysia/02_Timeseries/CPA_CPR/0_vars_timeseries/EVI/_shiftrev'
dir_list = ["A1","A2","A3","A4"] #
csv_dir_list = [csv_parent_dir + os.sep + a for a in dir_list]
# csv_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR"
# out_parent_dir = r"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels_EVI"
out_parent_dir = '/Volumes/PortableSSD/Malaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels_EVI'

start_date = '2000-01-01'
end_date = '2023-12-31'

## csvを全部読んでdfにする
variable_list = [
                  # "GOSIF",
                  "EVI",
                  "rain", 
                  "temp",
                  "VPD",
                  "Et",
                  "Eb",
                  "SMDSCE",
                 "SMDSC2",
                 "VODDSCE",
                 "VODDSC2"
                 ]


""" #Nodataの変換"""
##No dataはnanにする #np.whereだとインデックスやcolumn消えた
def convert_error(df, vari):
    ## GOSIF conversion here...
    if vari=="GOSIF":
        df_nan = df.where((df != 32767) | (df != 32766)) 
        df_nan = df_nan.where( df != 32768) #32768 looks ocean
        # see = df_nan.iloc[13330:15335,500:1000]
        df_nan = df_nan*0.0001
        df = df_nan
    if vari=="EVI":
        df_nan = df.where(df > 0) 
        # see = df_nan.iloc[13330:15335,500:1000]
        df_nan = df_nan*0.0001
        df = df_nan
    
    if vari=="SMDSCE" or vari=="SMDSC2" or vari=="VODDSCE" or vari=="VODDSC2":
        df_nan = df.where((df != 0)) 
        # see = df_nan.iloc[13330:15335,500:1000]
        df = df_nan
            
    df = df.astype("float16")
    # see2 = df.iloc[13330:15335,500:1000]
    df[df < 0] = np.nan
    
    return df


""" # 日付がインデックス、ピクセルインデックスが列名に整形, period絞る """
def sort_index_date(df):    
    df_T = df.T
    #Unnamedになった0行目削除
    df_T = df_T[1:]
    ## Indexの日付でソート
    df_T_reset = df_T.reset_index()
    df_T_reset["datetime"] = pd.to_datetime(df_T_reset['index'])
    df_T_reset = df_T_reset.sort_values('datetime')
    df_T_reset = df_T_reset.set_index("datetime")
    df_T_reset = df_T_reset.drop("index", axis=1)
    df_T_period = df_T_reset.loc[start_date:end_date]
    
    return df_T_period


""" # ignore nan when resample sum  """
def resample_sum(df, df_allnan):
    
    df_collist = []
    for col in df.columns:
        # col = df.columns[1]
        # df_col.iat[15] = 5
        # df_col.iat[20] = 5
        df_col = df[col]
        df_col_drop = df_col.dropna()
        df_col_monthly = df_col_drop.resample("M").sum()
        df_col_allmonth = pd.concat([df_allnan, df_col_monthly], axis=1)
        df_col_allmonth = df_col_allmonth.iloc[:,1:]
        df_collist.append(df_col_allmonth)
    
    df_sum = pd.concat(df_collist, axis=1)
    df_sum_sort = df_sum.sort_index(axis=1)
    
    return df_sum_sort



""" #処理 """
for csv_dir in csv_dir_list:
    PageName = os.path.basename(csv_dir)
    csvs = glob.glob(csv_dir+os.sep + "*.csv")
    # csvs = [c for c in csvs if "SMDSC" in c or "VODDSC" in c] #特定のcsvのとき →全部必要
    
    """ #処理 """
    monthly_dic = {}
    for variable in tqdm(variable_list):
        # variable = "Et"
        csvfile_var = [c for c in csvs if variable in c][0]
        print(csvfile_var)
        df_var = pd.read_csv(csvfile_var)
        # see = df_var.iloc[13330:15335,500:1000]
        df_var = convert_error(df_var, variable)  
        df_var_sort = sort_index_date(df_var)
        # seee = df_var_sort.iloc[3000:3040,13305:13307]
        # seee = df_var_sort.head()
        
        """ # prepare df only nan for later concat """ 
        df_nan = pd.DataFrame(index=df_var_sort.index)
        df_nan = df_nan.resample('M').sum()
        df_nan["nan"] = np.nan
                
        
        """ # monhly dataにする """
        # flux data to sum, amout data to mean
        # resample to igonore nan when mean, but sum produce 0        
        if variable=="rain" or variable=="Et" or variable=="Eb":
            df_var_monthly = resample_sum(df_var_sort, df_nan)
            # df_var_monthly = df_var_sort.resample('M').sum(numeric_only=True)
            # seee_monthly = df_var_monthly.head()
        else:
            df_var_monthly = df_var_sort.resample('M').mean(numeric_only=True)
            # seee_monthly = seee.resample('M').mean(numeric_only=True)
           
        monthly_dic[variable] = df_var_monthly
        
        for df in [df_var_sort, df_nan, df_var]:
            del df

    
    """#一列ずつ（1ピクセルずつ）取り出しconcat, 出力する """
    # len_list = [len(df_GPP_monthly),len(df_KBDI_monthly),len(df_rain_monthly),
    #             len(df_temp_monthly),len(df_Et_monthly),len(df_Eb_monthly),
    #             len(df_SM_monthly),len(df_VOD_monthly),
    #             ]
    
    # minindx = len_list.index(min(len_list))
    
    for idx in tqdm(range(monthly_dic["EVI"].shape[1])):
        # idx=13723
        pixel_list = []
        for variable, df in monthly_dic.items():
            # GPP_at_idx = df_GPP_monthly.loc[:,idx].rename("GPP", inplace=True)
            var_at_idx = df.loc[:,idx].rename(variable, inplace=True)
            pixel_list.append(var_at_idx)
        
        df_concat = pd.concat(pixel_list, axis=1)
        final_dir = out_parent_dir + os.sep + PageName
        os.makedirs(final_dir, exist_ok=True)
        outfile = final_dir + os.sep + f"{str(idx)}.csv"
        df_concat.to_csv(outfile)
    
    
    


        
       
    
        





