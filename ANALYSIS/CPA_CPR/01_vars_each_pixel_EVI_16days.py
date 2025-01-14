# -*- coding: utf-8 -*-
"""
PCA用にピクセルごとに目的変数と変数を時系列に並べたcsvを出力する。
rows*colmnsの数だけ出力される
EVIに合わせたTemporatl resolutionにする
"""

import os, sys
import glob
# import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
import time
from datetime import datetime
from datetime import datetime, timedelta
from tqdm import tqdm

pagename = sys.argv[1]
# csv_parent_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\0_vars_timeseries\EVI"
csv_parent_dir = '/Volumes/SSD_2/Malaysia/02_Timeseries/CPA_CPR/0_vars_timeseries/EVI/_shiftrev'
# dir_list = ["A1","A2","A3","A4"] #
dir_list = [pagename]
csv_dir_list = [csv_parent_dir + os.sep + a for a in dir_list]
# out_parent_dir = r"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels_EVI_16days"
out_parent_dir = '/Volumes/PortableSSD/Malaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels_EVI_16days'

start_date = '2000-01-01'
end_date = '2023-12-31'
years = [y for y in range(2000,2023+1)]

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
    df[df < 0] = np.nan #convert negative to nan here
    
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


""" # collect EVI dates """
sample_evi_csv = csv_dir_list[0] + os.sep + f"{pagename}_EVI_pixels_dates.csv"
df_evi_sample = pd.read_csv(sample_evi_csv)
df_evi_sample_datestr = df_evi_sample.columns.to_list()
df_evi_sample_date = [datetime.strptime(d, "%Y-%m-%d") for d in df_evi_sample_datestr if not d=="Unnamed: 0"]
df_evi_sample_date_sort = sorted(df_evi_sample_date)

evi_sample_dates = []
for i,d in enumerate(df_evi_sample_date_sort):
    start_d = df_evi_sample_date_sort[i]
    if not i==len(df_evi_sample_date_sort)-1:
        end_d = df_evi_sample_date_sort[i+1] - timedelta(days=1)
    else:
        end_d = datetime(2023, 12, 31, 0, 0)
    evi_sample_dates.append([start_d, end_d])
    
    

""" # ignore nan when resample sum  """
def resample_sum16days(df, variable):
    ## 16days from 1st Jan every year -> need alligned with EVI
    df_collist = []
    # for col in tqdm(df.columns):
    for col in tqdm(df.columns):
        # col = 12867 #A4,Et
        # col = 5000 #A1,Et
        # col = 13330 #A1,VODD2
        df_col = df[col]
        
        # df_col_years = []
        # for yr in years: #16days from 1st Jan
        #     df_col_yr = df_col[df_col.index.year==yr]
        #     # df_col_drop = df_col_yr.dropna()
        #     if variable == "rain" or variable=="Et" or variable=="Eb":
        #         df_col_monthly = df_col_yr.resample("16D").apply(lambda x: np.nan if x.isna().all() else x.sum())
        #     else:
        #         df_col_monthly = df_col_yr.resample("16D").apply(lambda x: np.nan if x.isna().all() else x.mean())
        
        """ #Allign with EVI"""
        df_col_years = []
        for s_e in evi_sample_dates:
            df_col_yr = df_col.loc[s_e[0]:s_e[1]]
            if variable == "rain" or variable=="Et" or variable=="Eb":
                df_col_val = df_col_yr.sum(skipna=True)
            else:
                df_col_val = df_col_yr.mean(skipna=True)
            df_col_monthly = pd.DataFrame({col: [df_col_val], "datetime":s_e[0]})
            df_col_monthly = df_col_monthly.set_index("datetime")
            df_col_years.append(df_col_monthly)
        
        df_col_allyears = pd.concat(df_col_years)
        ## 念のためソート
        df_col_allyears_reset = df_col_allyears.reset_index()
        df_col_allyears_sort = df_col_allyears_reset.sort_values('datetime')
        df_col_allyears_sort = df_col_allyears_sort.set_index("datetime")
        
        df_collist.append(df_col_allyears_sort)
    
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
        # variable = "VODDSC2"
        csvfile_var = [c for c in csvs if variable in c][0]
        print(csvfile_var)
        df_var = pd.read_csv(csvfile_var)
        # see = df_var.iloc[13330:15335,500:1000]
        df_var = convert_error(df_var, variable)  
        df_var_sort = sort_index_date(df_var)
        # seee = df_var_sort.head()
        # seee = df_var_sort.iloc[500:1000,13330:15335]
        
                        
        """ # 16days dataにする """
        # flux data to sum, amout data to mean
        # resample to igonore nan when mean, but sum produce 0
        df_var_monthly = resample_sum16days(df_var_sort, variable)
           
        monthly_dic[variable] = df_var_monthly
        
        for df in [df_var_sort, df_var]:
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
    
    
    


        
       
    
        





