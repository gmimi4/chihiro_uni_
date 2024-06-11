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
import statsmodels.formula.api as smf
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
palm_indx_txt = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index\palm_index_shape_72_203.txt"
p_val_tif = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_010\p_values_importance.tif"
importance_tif_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_010"
out_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\1_quadratic\p_010"

# Palmのあるインデックス取得
palm_idx_df = pd.read_csv(palm_indx_txt, header=None)
palm_idx_list = palm_idx_df.values.tolist()
palm_idx_list = [t[0] for t in palm_idx_list]

# p_valが有意なインデックス取得
p_src = rasterio.open(p_val_tif)
p_arr = p_src.read(1)
p_arr_1d = np.ravel(p_arr)
p_idx = list(np.where(p_arr_1d < 0.1)[0]) #0.05

#両方のインデックスの積集合
use_idx = list(set(palm_idx_list) & set(p_idx))

csvs = glob.glob(os.path.join(csv_dir,"*.csv"))
csvs_use = [c for c in csvs if int(os.path.basename(c)[:-4]) in use_idx]
# csvs_use = csvs

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

"""　#Observed dataのmean と STL resid和の組み合わせを得る """
# months = [m+1 for m in range(12)]
all_resid = {}
for csvfile in tqdm(csvs_use):    
    # csvfile = [c for c in csvs_use if "7144" in c][0] #4447 #8682 #1838 #7144
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
            # variable = 'temp'
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
        
        
""" # 変数ごとにまとめてdfにする"""
var_datasets = {}
for variable in vars_list:
    # variable="SM"
    resid_list=[]
    for i, residic in all_resid.items():
        resid_list.append(residic[variable])
    var_datasets[variable] = resid_list

""" # quadratic equation作成"""
# dfにする
for variable in vars_list:
    # variable="SM"
    target_var_dataset = var_datasets[variable]
    df_mean_vari = pd.DataFrame(target_var_dataset, columns=["mean","residualsum"])
    
    #まずプロット
    # fig,ax = plt.subplots()
    # ax.scatter(df_mean_vari["mean"], df_mean_vari["variance"])
    
    ## quadraticのfittingする
    degree = 2
    
    x = df_mean_vari["mean"]
    y = df_mean_vari["residualsum"]
    # df_mean_vari["meanSQ"] = df_mean_vari["mean"]**2
    
    weights = np.polyfit(x, y, degree)
    model = np.poly1d(weights)
    results = smf.ols(formula='residualsum ~ model(mean)', data=df_mean_vari).fit()
    # results = smf.ols(formula='variance ~ meanSQ + mean', data=df_mean_vari).fit()
    results.summary()
    pval = results.pvalues[1]
    # print(model)
    
    fig,ax = plt.subplots()
    # xmin = math.floor(x.min())
    # xmax = math.ceil(x.max())
    xmin = x.min()
    xmax = x.max()
    steps = int(10)
    polyline = np.linspace(xmin, xmax, steps)
    ax.scatter(x, y)
    ax.set_title(variable, fontsize=18)
    if pval < 0.1:
        ## Plot
        ax.plot(polyline, model(polyline), color="red", linewidth=2)
        coefs = list(weights)
    else: #p値が有意でない場合はvarianceの平均をとる
        vari_ave = y.mean()
        ax.axhline(vari_ave, color="red", linestyle='-', linewidth=2)
        coefs = [0, 0, vari_ave]
    figname = os.path.join(out_dir,f"{variable}_mean_resid_quadratic_deg{degree}.png")
    fig.savefig(figname)
    
    coefname = os.path.join(out_dir, f"coef_{variable}.txt")
    coefs_str = [str(c) for c in coefs]
    with open(coefname, "w") as coeffile:
        coeffile.write(",".join(coefs_str))



