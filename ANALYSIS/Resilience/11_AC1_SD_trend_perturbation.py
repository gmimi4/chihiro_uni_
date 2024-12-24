# -*- coding: utf-8 -*-
"""
# Empirical evidence for recent global shifts in vegetation resilience
"""

import numpy as np
import pandas as pd
import os,sys
import glob
from statsmodels.tsa.seasonal import STL
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import rasterio
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import scipy.stats, scipy.signal
from tqdm import tqdm
# os.chdir("/Users/wtakeuchi/Desktop/Python/ANALYSIS/Resilience") 
# import _01_pca_pcr_ARX_anomaly_EVI_recovery


pagename = sys.argv[1]
# pagename = 'A1'
in_dir = f'/Volumes/PortableSSD 1/MAlaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels_EVI_16days/{pagename}' #change later
out_dir_parent = '/Volumes/SSD_2/Malaysia/02_Timeseries/Resilience/07_perturbation'
out_dir = out_dir_parent + os.sep + f"{pagename}"
os.makedirs(out_dir,exist_ok=True)

csvs = glob.glob(in_dir + os.sep +'*.csv')

sample_tif = f"/Volumes/PortableSSD 1/MAlaysia/MODIS_EVI/01_MOD13A2061_resample/_4326_res01_age_adjusted/extent/MODEVI_20221016_4326_res01_adj_extentafter_{pagename}.tif"
with rasterio.open(sample_tif) as src:
    src_arr = src.read(1)
    meta = src.meta
    transform = src.transform
    height, width = src_arr.shape[0], src_arr.shape[1]
    # profile = src.profile
## 参照しているラスターのAffineのheight pixelはマイナスになっていてほしい
meta.update({"nodata":np.nan})


""" # STL from paper code"""
# f = 24
# ns = 7
def robust_stl(series, period=24, smooth_length=7):
    def nt_calc(f,ns):
        '''Calcualte the length of the trend smoother based on
        Cleveland et al., 1990'''
        nt = (1.5*f)/(1-1.5*(1/ns)) + 1 #Force fractions to be rounded up
        if int(nt) % 2. == 1:
            return int(nt)
        elif int(nt) % 2. == 0:
            return int(nt) + 1            
    def nl_calc(f):
        '''Calcualte the length of the low-pass filter based on
        Cleveland et al., 1990'''
        if int(f) % 2. == 1:
            return int(f)
        elif int(f) % 2. == 0:
            return int(f) + 1
    res = STL(series, period, seasonal=smooth_length, trend=nt_calc(period,smooth_length), low_pass=nl_calc(period), seasonal_deg=1, trend_deg=1, low_pass_deg=1, seasonal_jump=1,trend_jump=1, low_pass_jump=1, robust=True)
    return res.fit()


# ----------------------
""" # Paper Code with modification"""
# ----------------------
def calc_ar1(x):
    return np.corrcoef(x[:-1], x[1:])[0,1]

def sliding_window_calc(x, win_size=5): #5years
    numdata = (365 *win_size)/16
    var = np.empty(x.shape)
    var.fill(np.nan)
    ar1 = np.empty(x.shape[0]) #287 or 574
    ar1.fill(np.nan)
    half_window = int(numdata / 2) #win_size
    ln = x.shape[0]
    for i in range(half_window, ln - half_window):
        subset = x[i - half_window : i + half_window]
        try:
            ar1_ = calc_ar1(subset)
            var_ = np.nanvar(subset)
        except:
            ar1_ = np.nan
            var_ = np.nan
        ar1[i] = ar1_
        var[i] = var_
    ar1 = np.array(ar1)
    var = np.array(var)
    return ar1, var

def fourrier_surrogates(ts, ns):
    ts_fourier  = np.fft.rfft(ts)
    random_phases = np.exp(np.random.uniform(0, 2 * np.pi, (ns, ts.shape[0] // 2 + 1)) * 1.0j) #low=0.0, high=1.0, size=None
    ts_fourier_new = ts_fourier * random_phases
    new_ts = np.real(np.fft.irfft(ts_fourier_new))
    return new_ts

def score_at_pct(ts, tau, ns):
    tsf = ts - np.nanmean(ts) #Center on zero
    tsf = tsf[~np.isnan(tsf)] #Strip NaN (5-year rolling winndows leave NaN on either end of the TS)
    tlen = tsf.shape[0] #Get shape
    new_ser = fourrier_surrogates(tsf, ns) #Create surrogates via phase shifting #10000 dataset
    stat = np.zeros(ns)
    for i in range(ns):
        stat[i] = scipy.stats.kendalltau(range(tlen-1), new_ser[i,:], nan_policy='omit')[0] #Calc KT for shuffled series
    p = scipy.stats.percentileofscore(stat, tau) #tauはなんパーセントにいるか
    return p

# ----------------------
        
""" #liner trend"""
def linertrend(x):
    x = x[~np.isnan(x)]
    xrange = [i for i in range(len(x))]
    trend, intercept = np.polyfit(xrange, x, 1)
    return trend


months = [m+1 for m in range(12)]
win_size = 5 #years

AC_dic = {}
SD_dic = {}
p_AC_dic = {}
p_SD_dic = {}
for csvfile in tqdm(csvs):
    # csvfile = [c for c in csvs if "11790" in c][0] #10000
    filename = os.path.basename(csvfile)[:-4]
    df_csv = pd.read_csv(csvfile, index_col = 'datetime',parse_dates=['datetime'])
    ser_evi = df_csv["EVI"]
    ## interpolate and remove nan
    ser_evi = ser_evi.interpolate(method='linear')
    ser_evi = ser_evi.dropna()
    
    # # -------------------------
    # """ # Seddon's PCA Yt-1 in moving window"""
    # # -------------------------
    # AC1_pca_dic = {}
    
    # p_val = 0.1
    # numdata = (365 *win_size)/16
    # ar_pca = np.empty(ser_evi.shape[0]) #287 or 574
    # ar_pca.fill(np.nan)
    # half_window = int(numdata / 2) #win_size
    # ln = ser_evi.shape[0]
    # df_csv_reset = df_csv.reset_index()
    # # for i in tqdm(range(half_window, ln - half_window)):
    # for i in range(half_window, ln - half_window):
    #     subset = df_csv_reset[i - half_window : i + half_window] #all var
    #     subset = subset.set_index("datetime")
    #     try:
    #         ar1_pca_val = _01_pca_pcr_ARX_anomaly_EVI_recovery.main(subset, p_val, 2000, 2023, 1) #timelag=1
    #         ar1_pca_val = ar1_pca_val["EVIt-1"]
    #     except:
    #         ar1_pca_val = np.nan

    #     ar_pca[i] = ar1_pca_val
        
    try:
        # -------------------------
        """ # STL for residual"""
        # -------------------------
        ser_fit = robust_stl(ser_evi)
        # ser_fit.plot()
        ser_resid = ser_fit.resid
    
        # -------------------------
        """ # Calc AC1 and SD"""
        # -------------------------  
        ar, sd = sliding_window_calc(ser_resid, 5)
        
        tau_ar = scipy.stats.kendalltau(range(len(ar)), ar, nan_policy='omit')[0]
        tau_var = scipy.stats.kendalltau(range(len(sd)), sd, nan_policy='omit')[0]
        
        try:
            pval_ar = score_at_pct(ar, tau_ar, ns=1000)
        except:
            pval_ar = np.nan
        try:
            pval_var = score_at_pct(sd, tau_var, ns=1000)
        except:
            pval_var = np.nan

        
        
        p_AC_dic[filename] = pval_ar
        p_SD_dic[filename] = pval_var
        
        # if pval_ar >=90 or pval_ar <=10:
        #     AC_dic[filename] = linertrend(ar)
        # else:
        #     AC_dic[filename] = np.nan
        # if pval_var >=90 or pval_var <=10:
        #     SD_dic[filename] = linertrend(sd)
        # else:
        #     SD_dic[filename] = np.nan
        
        AC_dic[filename] = linertrend(ar)
        SD_dic[filename] = linertrend(sd)
    
    except:
        p_AC_dic[filename] = np.nan
        p_SD_dic[filename] = np.nan
        AC_dic[filename] = np.nan
        SD_dic[filename] = np.nan
        
                    
                   
""" Export tif"""
def to_raster(tar_dic, outrasname):
    ras_dic = {}
    for i,numval in tar_dic.items():
        # i=4446
        # imoprtance_dic = num_dic[str(i)]
        ras_dic[int(i)] = numval
    
    #念のためindx順にソート
    ras_dic_sort = sorted(ras_dic.items())
    
    #これに入れる
    importance_arr = np.full(len(ras_dic_sort), np.nan)
    for i in ras_dic_sort:
        arri = i[0]
        arrval = i[1]
        np.put(importance_arr, [arri], arrval)
    # reshape
    importance_arr_re = importance_arr.reshape((height, width))
    
    out_file = out_dir + os.sep +f"{pagename}_{outrasname}.tif"
    with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
        with rasterio.open(out_file, 'w', **meta) as dst:
          dst.write(importance_arr_re, 1)


to_raster_dic = {"pval_AC":p_AC_dic,
                 "pval_sd":p_SD_dic,
                 "AC":AC_dic,
                 "sd":SD_dic,
                 }


for outrasname, tar_dic in to_raster_dic.items():
    to_raster(tar_dic, outrasname)
        
    
    
    
    
    
    
    
    
    
        
 
    