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
import statsmodels.formula.api as smf
import math
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose
import copy
import warnings
import rasterio
from rasterio.plot import show
from rasterio.crs import CRS
from pyproj.crs import CRS
with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
    rio_crs = CRS.from_epsg(4326)
    proj_crs = CRS.from_user_input(rio_crs)

csv_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\1_vars_at_pixels"
# palm_indx_txt = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index\palm_index_shape_72_203.txt"
p_val_tif = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_010\p_values_importance.tif"
importance_tif_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_010"
out_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\1_stl\p_010"
coef_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\1_quadratic\p_010"

# Palmのあるインデックス取得
# palm_idx_df = pd.read_csv(palm_indx_txt, header=None)
# palm_idx_list = palm_idx_df.values.tolist()
# palm_idx_list = [t[0] for t in palm_idx_list]

# p_valが有意なインデックス取得
p_src = rasterio.open(p_val_tif)
p_arr = p_src.read(1)
p_arr_1d = np.ravel(p_arr)
p_idx = list(np.where(p_arr_1d < 0.1)[0]) #0.05

#両方のインデックスの積集合
# use_idx = list(set(palm_idx_list) & set(p_idx))

csvs = glob.glob(os.path.join(csv_dir,"*.csv"))
# csvs_use = [c for c in csvs if int(os.path.basename(c)[:-4]) in use_idx]
csvs_use = csvs

nps=12
n_s =13
""" # STL plot def"""
# parameters: https://mlpills.dev/time-series/time-series-forecasting-with-stl/
def plot_STL(df): #periodは日付indexから自動で決められる
  df_int = df.interpolate() #defaultで線形補間
  df_intna = df_int.dropna()
  stl = STL(df_intna, period=nps, seasonal=n_s)
  res_stl = stl.fit()
  # fig = res_stl.plot()
  
  return res_stl

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
   

""" #処理開始　STLのresidualsを得る""" 
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
        try: #columnにDSCとASCがある場合
            df_csv["SM"] = df_csv[["SMASC", "SMDSC"]].mean(skipna=True, axis='columns')
            df_csv["VOD"] = df_csv[["VODASC", "VODDSC"]].mean(skipna=True, axis='columns')
            df_csv= df_csv.drop(["SMASC","SMDSC","VODASC","VODDSC"], axis=1)
        except: #片方だけの場合
            vars_list = df_csv.columns.tolist()
            sv_col_list = [c for c in vars_list if "SM" in c or "VOD" in c]
            for colname in sv_col_list:
                newname = colname[:-3]
                df_csv= df_csv.rename(columns={colname: newname})
        
        vars_list = df_csv.columns.tolist()
        
        """ #raw月平均でSTLする"""     
        df_csv_z = df_csv.copy()
        
        var_resid = {}
        for variable in  vars_list:       
            df_var = df_csv_z.loc[:,variable]
            stl_result = plot_STL(df_var)
            # fig = stl_result.plot()
            df_residus = stl_result.resid #residual
            df_residus_abs = df_residus.abs() #絶対値にする
            sum_residus = df_residus_abs.sum()
            
            # """# seasonl meanが欲しい""" #→seasonalってプラスマイナスだった
            # df_seasonal = stl_result.seasonal #seasonal
            # seasonal_mean = df_seasonal.mean() #全seasonalのmean
            # # monthly_mean_seasonl = monthly_mean_dic(df_seasonal) #seasonalの月別mean→使わずか
            
            """# observationのmean"""
            df_obs = stl_result.observed
            obs_mean = df_obs.mean()
            
            # """# residualをratioで得る"""
            # var_resid[variable] = sum_residus/seasonal_mean
            # var_resid[variable] = sum_residus/obs_mean
            """# residualのsumを得る"""
            var_resid[variable] = [obs_mean, sum_residus]
            
    
        all_resid[int(filename)] = var_resid
    else: #有意pではないピクセル
        all_resid[int(filename)] = {}
        


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




