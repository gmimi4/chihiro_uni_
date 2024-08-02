# -*- coding: utf-8 -*-
"""
# Divide simpley half period
# try ADF or KPSS for stochastic trend
# https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
"""

import numpy as np
import pandas as pd
import os, sys
import glob
import time
import matplotlib.pyplot as plt  # for display purposes
import ruptures as rpt
import math
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.seasonal import STL #test
from scipy.fft import fft, ifft, fftfreq
from scipy import fftpack
from scipy import stats
import copy
import rasterio
from tqdm import tqdm

# csvfile = "/Volumes/PortableSSD/Malaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels/A1/15706.csv"
# page = "A1"
page = sys.argv[1]
csv_dir = rf"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels\{page}"
startyear = 2002
endyear = 2022
out_dir = r"D:\Malaysia\02_Timeseries\Resilience\03_halfperiod"

csvs = glob.glob(csv_dir + os.sep + "*.csv")

sample_tif = rf'F:/MAlaysia/SIF/GOSIF/02_tif_age_adjusted/res_01/extent/GOSIF_2000081_extent_adj_res01_extent_{page}.tif'
with rasterio.open(sample_tif) as src:
    src_arr = src.read(1)
    meta = src.meta
    transform = src.transform
    height, width = src_arr.shape[0], src_arr.shape[1]
    # profile = src.profile
    
## 参照しているラスターのAffineのheight pixelはマイナスになっていてほしい
meta.update({"nodata":np.nan})


results = {
    # "NumChangePoints":{},
           "trendDiff":{},
           "stdDiff":{},
           "amplitudeDiff":{}}

for csvfile in tqdm(csvs):
    # 7460 error
    # csvfile = r"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels\A1\7443.csv" #15706 no changep #7443
    df_csv = pd.read_csv(csvfile, index_col = 'datetime',
                         parse_dates=['datetime'])
    
    filename = os.path.basename(csvfile)[:-4]

    """ # AMSREとAMSR2のギャップを補正する"""   
    pi_e = df_csv.loc[:,["SMDSCE", "VODDSCE"]]
    pi_2 = df_csv.loc[:,["SMDSC2", "VODDSC2"]]
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
    
    """ # extract SIF"""
    df_csv_sif = df_csv.loc[:, "GOSIF"]
    
    """ # z scoring"""
    seri_csv_sif_z = df_csv_sif.copy()
    df_csv_sif_z = seri_csv_sif_z.to_frame()
    sif_mean = df_csv_sif_z["GOSIF"].mean()
    sif_std = df_csv_sif_z["GOSIF"].std()
    df_csv_sif_z.loc[:, "GOSIFz"] = (df_csv_sif_z["GOSIF"]-sif_mean)/sif_std
    
    
    data_sif = df_csv_sif_z["GOSIFz"].values #some GOSIF pixels not exist in early years
    
    nan_chck = data_sif[~np.isnan(data_sif)]
    
    if len(nan_chck) ==0: # no valid area
        for ds in results.values():
            ds[filename]=np.nan
        continue
        
    else: # there are valid values
        
        """ # half period as change date"""
        my_bkps_fin = [int(len(data_sif)/2)] #set half period

        change_dates = df_csv_sif[my_bkps_fin].index.tolist()
        
        ### collect num of change point, dates -> あとでreturnする
        change_list = [len(my_bkps_fin), change_dates]
        
        
        # df_csv_sif_reset = df_csv_sif.reset_index() # for check
        
        """ # divide data by change points """
        # obtain index poisitons of change points
        # current_position = df_csv_sif.index.get_loc(change_dates[0])    
        my_bkps_divi = []
        for i in range (len(my_bkps_fin)):
            if i ==0:
                my_bkps_divi.append([0, my_bkps_fin[i]-1])
    
            else: 
                my_bkps_divi.append([my_bkps_fin[i-1], my_bkps_fin[i]-1])
    
            if i == len(my_bkps_fin)-1:
                my_bkps_divi.append([my_bkps_fin[i], len(data_sif)-1])
            else:
                continue
    
        
        """　# Set data length as the shortest one and take last months for later segment """
        my_bkps_length = [m[1] - m[0] +1 for m in my_bkps_divi]
        shortest_length = min(my_bkps_length)
        shortest_indx = my_bkps_length.index(shortest_length)
        
        my_bkps_divi_rev = copy.deepcopy(my_bkps_divi)
        for i,ilst in enumerate(my_bkps_divi_rev):
            if i != shortest_indx:
                end = ilst[1]
                start_update = end - shortest_length 
                ilst[0] = start_update #employ latest period
        
        
        """ extract first and last period """
        my_bkps_divi_rev = [my_bkps_divi_rev[0], my_bkps_divi_rev[-1]]
                    
        amps = []
        stds = []
        trds = []
        """ # divide at change poisition """    
        for ilst in my_bkps_divi_rev: #my_bkps_divi
            df_divi = data_sif[ilst[0]:ilst[1]+1]
            
            ### nanは外挿補完する
            df_divi_valid = df_divi.interpolate() #nan線形補完
            ## drop nan completely
            df_divi_valid = df_divi_valid[~np.isnan(df_divi_valid)]
            
            # not all nan or 0                                
            if (len(df_divi_valid[~np.isnan(df_divi_valid)]) != 0) and (len(np.where(df_divi_valid!=0)[0]) != 0):
                
                """　#Extract trend by ADF """
                dftest = adfuller(df_divi_valid, autolag="AIC")
                dfoutput = pd.Series(
                    dftest[0:4],
                    index=[
                        "Test Statistic",
                        "p-value",
                        "#Lags Used",
                        "Number of Observations Used",
                    ],
                )
                for key, value in dftest[4].items():
                    dfoutput["Critical Value (%s)" % key] = value #Series
                
                pval_adf = dfoutput["p-value"]
                
                
                ### """　# Detrend by differencing ("*_diff")""" -> not employed
                ## convert series to dataframe
                # df_divi_frame = df_divi_valid.to_frame()
                
                # df_divi_frame["ADF_diff"] = df_divi_frame["GOSIF"] - df_divi_frame["GOSIF"].shift(1)
                # df_divi_frame["ADF_diff"].dropna().plot(figsize=(12, 8))
                
               
                """　#Extract trend and seasonality by moving average """
                ## stasmodel
                stats_result = seasonal_decompose(df_divi_valid, period=12, two_sided=True) #
                stats_trend = stats_result.trend ## statsmodel
                stats_seasonal = stats_result.seasonal #it seems average after detrended
                # stats_seasonal = stats_result.observed - stats_trend # test for pure subtraction
                
                
                # """ # ADF """ -> not employed
                # stats_seasonal = df_divi_frame["ADF_diff"]
                # from pandas.core.nanops import nanmean as pd_nanmean
                # period =12
                # period_ave = np.array([pd_nanmean(stats_seasonal[i::period], axis=0) for i in range(period)])
                # stats_seasonal_ave = np.tile(period_ave.T, len(stats_seasonal) // period + 1).T[:len(stats_seasonal)]
                
                """
                fig = stats_result.plot()
                fig.set_size_inches((10, 5))
                ## if extract seasonal for plotting
                # fig, ax = plt.subplots(figsize=(10, 5))
                # stats_seasonal.plot(ax=ax)                
                fig.tight_layout()
                
                out_dir_fin = out_dir + os.sep + "_png"
                outfilename = out_dir_fin + os.sep + f"timeplot_{filename}_{ilst[0]}-{ilst[1]}.png"
                # outfilename = out_dir_fin + os.sep + f"timeplot_{filename}_{ilst[0]}-{ilst[1]}_ADFdiff.png"
                fig.savefig(outfilename)
                """
                
                """# fourier transform for seasonality"""
                # サンプル数
                stats_seasonal_vali = stats_seasonal[~np.isnan(stats_seasonal)]
                N = len(stats_seasonal_vali)
                
                # フーリエ変換する
                y = stats_seasonal_vali.values
                yf = fft(y)
                
                # 周波数軸の作成
                # サンプリング間隔
                # T = 1
                T = 1/12
                xf = fftfreq(N, T)[:N//2] #T: sampling space
                
                """
                fig = plt.figure()
                plt.plot(xf, 2.0/N * np.abs(yf[0:N//2])) ## ここで2/Nやってた
            
                out_dir_fin = out_dir + os.sep + "_png"
                # outfilename = out_dir_fin + os.sep + f"ampli_freq_{filename}_{ilst[0]}-{ilst[1]}.png"
                outfilename = out_dir_fin + os.sep + f"ampli_freq_{filename}_{ilst[0]}-{ilst[1]}_ADFdiff.png"
                fig.savefig(outfilename)
                """
                
                """# Amplitude square sum"""
                amplid = sum(2.0/N * np.abs(yf[0:N//2]))
                amps.append(amplid)
                
                """# simply SD """
                sd = np.std(df_divi_valid)
                stds.append(sd)
                
                # """# trend using trend data """ # -> not well clear or worked
                # stats_trend_reset = stats_trend.reset_index()
                # stats_trend_reset_nan = stats_trend_reset[~np.isnan(stats_trend_reset.trend)]
                # tr_x = stats_trend_reset_nan.index.tolist() # stats_trend already divided
                # tr_y = stats_trend_reset_nan.trend.values.tolist()
                
                # result_sq = stats.linregress(tr_x, tr_y)
                # result_slp = result_sq.slope
                # trds.append(result_slp)
                
                """# trend using average of the period """
                if pval_adf > 0.1:
                    df_divi_valid_ave = np.nanmean(df_divi_valid)
                    trds.append(df_divi_valid_ave)
                else:
                    trds.append(0)
                
            
            else: #all nan or 0
                amps.append(np.nan) #differene below becomes nan
                stds.append(np.nan)
                trds.append(np.nan)
                
                
            
        """# get difference or last trend """
        sd_diff = stds[1] -stds[0] #after -before: if increase, positive
        amp_diff = amps[1] - amps[0]
        trd_diff = trds[1] - trds[0]
        
        results["stdDiff"][filename] = sd_diff
        results["amplitudeDiff"][filename] = amp_diff
        results["trendDiff"][filename] = trd_diff # minus if decreasing
        

""" # Export to tif"""
for ky, valdict in results.items():
        
    #念のためindx順にソート
    valdict_sort = sorted(valdict.items()) #タプルになった(indx, importance)    
    #これに入れる
    importance_arr = np.full(len(valdict_sort), np.nan)
        
    for i in valdict_sort:
        arri = i[0]
        arrval = i[1]
        np.put(importance_arr, [arri], arrval)
        
    # reshape
    importance_arr_re = importance_arr.reshape((height, width))
    
    out_file = out_dir + os.sep +f"{page}_{ky}_half.tif"
    with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
        with rasterio.open(out_file, 'w', **meta) as dst:
          dst.write(importance_arr_re, 1)  
        
    
    
    
