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
import statsmodels.formula.api as smf
import math
from statsmodels.tsa.seasonal import STL
# from statsmodels.tsa.seasonal import seasonal_decompose
import copy
import warnings
import rasterio
from rasterio.plot import show
from rasterio.crs import CRS
from pyproj.crs import CRS
with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
    rio_crs = CRS.from_epsg(4326)
    proj_crs = CRS.from_user_input(rio_crs)

PageName = 'A4'
# csv_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\1_vars_at_pixels"
csv_dir = f'/Volumes/PortableSSD/MAlaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels/{PageName}'
p_val_tif = f'/Volumes/SSD_2/Malaysia/02_Timeseries/CPA_CPR/2_out_ras/p_01/{PageName}_p_values_importance_2013-2022.tif'
importance_tif_dir = os.path.dirname(p_val_tif)
out_dir = '/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/1_std/std_ras/p_01'
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


""" # dfからMonthly mean dicを得る"""
months = [m for m in range(1,13)]
def monthly_mean_dic(df):
    #df=df_seasonal
    month_dic={}
    for m in months:
        month_row = df[df.index.month == m]
        monthly_mean = month_row.mean()
        month_dic[m] = monthly_mean
    
    return month_dic


""" # detrend data, and obtain cv for each month, then sum up"""
all_resid = {}
for csvfile in tqdm(csvs_use):    
    # csvfile = [c for c in csvs_use if "13330" in c][0] #4447 #8682
    # csvfile = r"D:\Malaysia\02_Timeseries\CPA_CPR\1_vars_at_pixels\1838.csv"
    filename = os.path.basename(csvfile)[:-4]
    df_csv = pd.read_csv(csvfile, index_col = 'datetime',
                         parse_dates=['datetime'])

    """ #SMとVODは平均にする"""
    #skipnaでいけた
    df_csv["SM"] = df_csv[["SMDSCE", "SMDSC2"]].mean(skipna=True, axis='columns')
    df_csv["VOD"] = df_csv[["VODDSCE", "VODDSC2"]].mean(skipna=True, axis='columns')
    df_csv= df_csv.drop(["SMDSCE","SMDSC2","VODDSCE","VODDSC2"], axis=1)
    
    vars_list = df_csv.columns.tolist()
    
    """ #月平均でdetrendする""" 
    monthly_mean_dic = monthly_mean_dic(df_csv)
    
    df_csv_de = df_csv.copy()
    
    for var in vars_list:
        df_csv_de[f"{var}de"] = np.nan
        df_var = df_csv_de.loc[:,var]
        month_dic = {}
        for m in months:
            month_mean = monthly_mean_dic[m][var]
            specific_month_rows = df_var[df_var.index.month == m]
            detrend = specific_month_rows - month_mean
            monthly_mean = detrend.mean()
            monthly_std = detrend.std()
            month_dic[m] = [monthly_mean, monthly_std]
            
            ##インデックスで抽出して元のデータフレームの新規列にzscoreを入れる
            specific_month_idx = specific_month_rows.index.tolist()
            df_csv_z.loc[specific_month_idx, f"{var}z"] = (df_csv_z[var]-monthly_mean)/monthly_std
    
    
    
    """ #raw月平均でSTLする"""     
    df_csv_z = df_csv.copy()
    
    var_resid = {}
    for variable in  vars_list:       
        df_var = df_csv_z.loc[:,variable]
        var_mean = df_var.mean()
        var_std = df_var.std()
        var_cv = var_mean / var_std
        
        var_resid[variable] = var_cv
        

    all_resid[int(filename)] = var_resid

        


""" # Export tif for cv of each variale """ 
with rasterio.open(tifs[0]) as src:
    arr = src.read(1)
    profile=src.profile
    height, width = arr.shape[0],arr.shape[1]
    
#念のためsort
all_resid_sort = sorted(all_resid.items())

for variable in  vars_list: 
    var_cv_arr = np.array([c[1][variable] for c in all_resid_sort]) #順番通りに取り出しているはず 
    var_cv_reshape = var_cv_arr.reshape((height, width)) 

    outfile = os.path.join(out_dir,"cv_{variable}.tif")
    with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
        with rasterio.open(outfile, "w", **profile) as dst:
            dst.write(var_cv_reshape, 1)










""" old """
""" # obtain climate weights of the pixel"""
### まずimportance tifsのarrayセットを得る ------------
tifs = glob.glob(os.path.join(importance_tif_dir, "*.tif"))
tifs = [t for t in tifs if "p_values" not in t]
vars_list2 = [os.path.basename(v)[:-4].split("_")[0] for v in tifs]

importance_dic = {}
for tif in tifs: #impmortance tifs
    varname = os.path.basename(tif)[:-4].split("_")[0]
    with rasterio.open(tif) as src:
        arr = src.read(1)
        arr_flat = arr.ravel()
        importance_dic[varname] = arr_flat


### weightsにする ------------------
# ピクセル内のcoefでStandarized -----
coefs = {} #ピクセルごとのcoefを回収する
for i in tqdm(range(len(arr_flat))):
    pixel_coefs={}
    for variable in vars_list2:
        varcoef = importance_dic[variable][i]
        pixel_coefs[variable]=varcoef
    coefs[i] = pixel_coefs

# 回収したcoefをピクセル内でnormalize -> ratioにする
coefs_scale = copy.deepcopy(coefs)
for i, coefdic in tqdm(coefs_scale.items()):
    # coefdic = coefs[4447]
    # coefdic = coefs[i]
    coeflist = [abs(c) for var, c in coefdic.items()] #絶対値
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try: #coeflistが全部nanだとWarningが出る
            maxcoef = np.nanmax(coeflist)
            mincoef = np.nanmin(coeflist)
            sumcoef = sum([abs(c) for c in coeflist])
            for variable in vars_list2:
                coef = abs(coefdic[variable])
                # coefscaled = (coef - mincoef) / (maxcoef-mincoef)
                coefscaled = coef / sumcoef
                coefdic[variable] =coefscaled
        except ValueError as e:
            for variable in vars_list2:
                coefdic[variable] =np.nan
    
    coefs_scale[i] = coefdic
   

""" #処理開始　CVを得る""" 
# months = [m+1 for m in range(12)]
all_resid = {}
for csvfile in tqdm(csvs_use):    
    # csvfile = [c for c in csvs_use if "8682" in c][0] #4447 #8682
    # csvfile = r"D:\Malaysia\02_Timeseries\CPA_CPR\1_vars_at_pixels\1838.csv"
    filename = os.path.basename(csvfile)[:-4]
    if int(filename) in p_idx:
        df_csv = pd.read_csv(csvfile, index_col = 'datetime',
                             parse_dates=['datetime'])
    
        """ #SMとVODは平均にする"""
        #skipnaでいけた
        try: #AMSREとAMSR2を一列にする
            df_csv["SM"] = df_csv[["SMDSCE", "SMDSC2"]].mean(skipna=True, axis='columns')
            df_csv["VOD"] = df_csv[["VODDSCE", "VODDSC2"]].mean(skipna=True, axis='columns')
            df_csv= df_csv.drop(["SMDSCE","SMDSC2","VODDSCE","VODDSC2"], axis=1)
        except: #片方だけの場合
            vars_list = df_csv.columns.tolist()
            sv_col_list = [c for c in vars_list if "SM" in c or "VOD" in c]
            for colname in sv_col_list:
                if "SM" in colname:
                    newname = "SM"
                elif "VOD" in colname:
                    newname = "VOD"
                else:
                    pass
                # newname = colname[:-3]
                df_csv= df_csv.rename(columns={colname: newname})
        
        vars_list = df_csv.columns.tolist()
        
        """ #raw月平均でSTLする"""     
        df_csv_z = df_csv.copy()
        
        var_resid = {}
        for variable in  vars_list:       
            df_var = df_csv_z.loc[:,variable]
            var_mean = df_var.mean()
            var_std = df_var.std()
            var_cv = var_mean / var_std
            
            var_resid[variable] = var_cv
            
    
        all_resid[int(filename)] = var_resid
    else: #有意pではないピクセル
        all_resid[int(filename)] = {}
        


""" # Export tif for cv of each variale """ 
with rasterio.open(tifs[0]) as src:
    arr = src.read(1)
    profile=src.profile
    height, width = arr.shape[0],arr.shape[1]
    
#念のためsort
all_resid_sort = sorted(all_resid.items())

for variable in  vars_list: 
    var_cv_arr = np.array([c[1][variable] for c in all_resid_sort]) #順番通りに取り出しているはず 
    var_cv_reshape = var_cv_arr.reshape((height, width)) 

    outfile = os.path.join(out_dir,"cv_{variable}.tif")
    with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
        with rasterio.open(outfile, "w", **profile) as dst:
            dst.write(var_cv_reshape, 1)
        
        


""" # Quadratic lineの係数収集 """        
## quadratic coef　dicの取得
qcoeftxts = glob.glob(coef_dir +os.sep +'*.txt')
qcoef_df_dic ={}
for txt in qcoeftxts:
    variable = os.path.basename(txt)[:-4].split("_")[1]   
    df = pd.read_csv(txt, header=None, sep=",").rename(columns={0:"x2",1:"x",2:"b"})
    qcoef_df_dic[variable] = df


""" # ピクセルのresiduals　と　Quadratic lineとのオフセットを得る """ 
def get_predicion(df, meanval):
    a = df.at[0,"x2"]
    b = df.at[0,"x"]
    c = df.at[0,"b"]
    prediction = a*meanval**2 + b*meanval + c
   
    return prediction
    
all_resid_offset = {}
for k,v in tqdm(all_resid.items()):
    # k=4447
    # v=all_resid[k]
    if len(all_resid[k]) == 0: #空リストの場合
        all_resid_offset[k] = {}
        continue
    else:
        offset_dic = {}
        for variable in vars_list:
            qcoef_df = qcoef_df_dic[variable]
            obmean = v[variable][0]
            prediction_variable = get_predicion(qcoef_df, obmean)
            difference = v[variable][1] - prediction_variable #STL residualが小さければマイナス
            offset_dic[variable] =difference    
    
    all_resid_offset[k] = offset_dic
    

     
"""##変数ごとにStandarizedする (全サイトでnormalize) """
## 変数ごとにresdualsを集める flatにする
var_resid_flat = {}
for variable in vars_list:
    resid_list = []
    for k,v in all_resid_offset.items():
        if len(all_resid_offset[k]) == 0: #空リストの場合
            continue
        else:
            resid = v[variable]
            resid_list.append(resid)
            
    var_resid_flat[variable] = resid_list


## 各変数のmaxとminを得る
max_min_dic = {}
for variable in vars_list:
    reslist = var_resid_flat[variable]
    maxres = max(reslist)
    minres = min(reslist)
    max_min_dic[variable] = [maxres, minres]
    
## offsetを0-1にStandarizedする
all_resid_01 = copy.deepcopy(all_resid_offset)

for variable in vars_list:
    varmax, varmin = max_min_dic[variable][0], max_min_dic[variable][1]
    
    for k,v in all_resid_offset.items():
        if len(v) == 0: #空リストの場合
            continue
        else:
            resid = v[variable]
            resid_01 = (resid - varmin)/(varmax - varmin)
            v[variable] = resid_01
        
        all_resid_01[k] = v #置換

#check
# vs = [v["GPP"] for i,v in all_resid_01.items() if len(v)>0]
    
"""　# Sensitivityの計算 """
# log10(ratio) * weight
# vars_list2 = [v for v in vars_list if v != "GPP"]

all_sensitivity = {}
for k,v in all_resid_01.items():
    # k=11126 #めちゃ小さいresidualを含みinfになる
    # v=all_resid_01[k]
    if len(v) == 0: #空リストの場合
        all_sensitivity[k] = np.nan
        continue
    else:
        # pixel_weights = obtain_importances_dic(k, vars_list2) #weigts取得
        pixel_weights = coefs_scale[k]
        sensitivity_ele =[]
        for variable in vars_list2:
            resid_01 = v[variable]
            with np.errstate(divide='raise'):                
                try:
                    # ratio = math.log10(v["GPP"]/resid_01) #zeroでない限り計算する
                    ratio = v["GPP"]/resid_01 #zeroでない限り計算する
                    byweight = ratio * pixel_weights[variable]
                    # if resid_01 > 0.0001: #めちゃ小さすぎる場合は計算しない
                    #     ratio = math.log10(v["GPP"]/resid_01)
                    #     byweight = ratio * pixel_importances[variable]
                    # else:
                    #     byweight = np.nan
                except:
                    byweight = np.nan
                    # print(k)
                
            sensitivity_ele.append(byweight)
        ##sum up # nanがあったら諦め
        sensitivity = np.sum(sensitivity_ele)
        
    all_sensitivity[k] = sensitivity



"""　# tifのExport """
#念のためsort
all_sensitivity_sort = sorted(all_sensitivity.items())
all_sensitivity_arr = np.array([t[1] for t in all_sensitivity_sort]) #順番通りに取り出しているはず 

with rasterio.open(tifs[0]) as src:
    arr = src.read(1)
    profile=src.profile
    height, width = arr.shape[0],arr.shape[1]


all_sensitivity_reshape = all_sensitivity_arr.reshape((height, width)) 

outfile = os.path.join(out_dir,"sensitivity_stl_quadratic.tif")
with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
    with rasterio.open(outfile, "w", **profile) as dst:
        dst.write(all_sensitivity_reshape, 1)

#plot
# ras = rasterio.open(outfile)
# show(ras)




