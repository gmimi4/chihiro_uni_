# -*- coding: utf-8 -*-
"""
# 
"""

import numpy as np
import pandas as pd
import os,sys
import glob
from statsmodels.tsa.seasonal import STL
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import rasterio
from tqdm import tqdm


pagename = sys.argv[1]
pagename = 'A1'
in_dir = f'/Volumes/PortableSSD/MAlaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels_EVI_16days/{pagename}'
out_dir_parent = '/Volumes/SSD_2/Malaysia/02_Timeseries/Resilience/07_perturbation'
out_dir = out_dir_parent + os.sep + f"{pagename}"
os.makedirs(out_dir_parent,exist_ok=True)

csvs = glob.glob(in_dir + os.sep +'*.csv')

sample_tif = f"/Volumes/PortableSSD/MAlaysia/MODIS_EVI/01_MOD13A2061_resample/_4326_res01_age_adjusted/extent/MODEVI_20221016_4326_res01_adj_extentafter_{pagename}.tif"
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


""" # Window length for differences (18points =9monts in paper)"""
def moving_diff(series, window = 18):
    half_window = window // 2
    
    series_reset = series.reset_index()
    series_reset_drop = series_reset.drop("datetime",axis=1)
    rolling_diff = (
        series_reset_drop.rolling(window=window, center=True) #center: input result in the middle
        .apply(lambda x: x[half_window:].mean() - x[:half_window].mean(), raw=True) #after - before
        # .apply(lambda series_reset: series_reset[:half_window].mean() - series_reset[half_window:half_window*2].mean(), raw=True)
    ) #ok
    #check
    # series_reset[:half_window].mean() - series_reset[half_window:half_window*2].mean()
    # series_reset[1:half_window+1].mean() - series_reset[half_window+1: half_window+1 + half_window].mean()
    return rolling_diff, series_reset
        
    
num_dic ={}       
for csvfile in tqdm(csvs):
    # csvfile = [c for c in csvs if "10000" in c][0]
    filename = os.path.basename(csvfile)[:-4]
    df_csv = pd.read_csv(csvfile, index_col = 'datetime',parse_dates=['datetime'])
    ser_evi = df_csv["EVI"]
    ## interpolate and remove nan
    ser_evi = ser_evi.interpolate(method='linear')
    ser_evi = ser_evi.dropna()
    
    try:
        # -------------------------
        """ # STL for residual"""
        # -------------------------
        ser_fit = robust_stl(ser_evi)
        # ser_fit.plot()
        ser_resid = ser_fit.resid
    
        # -------------------------
        """ # Moving difference"""
        # -------------------------  
        seri_movingdiff, series_reset = moving_diff(ser_resid)
        
        # -------------------------
        """ # Savitzky-Golay filter"""
        # -------------------------
        seri_sgfilt = savgol_filter(seri_movingdiff.resid, window_length=7, polyorder=1)
        
        # fig, ax = plt.subplots(figsize=(10,5))
        # ax.plot(series_reset.datetime, seri_movingdiff.resid, label='original')
        # ax.plot(series_reset.datetime, seri_sgfilt, c='red', label='savgol_filter')
        # ax.set_ylabel('EVI residulas')
        # ax.legend(loc='best')
        # fig_dir = out_dir + os.sep + "_png"
        # os.makedirs(fig_dir,exist_ok=True)
        # fig.savefig(fig_dir + os.sep + f'SavitzkyGoley_{filename}.png')
        # plt.show()
        
        # -------------------------
        """ # Pick 99 percentile -> maybe pick negative change"""
        # -------------------------
        # contat with datetime
        df_sgfit = pd.DataFrame({"sgfilt":seri_sgfilt})
        df_sgfit = pd.merge(series_reset, df_sgfit, left_index=True, right_index=True, how='inner')
        df_sgfit = df_sgfit.set_index("datetime")
        
        sgfilt_vals = df_sgfit.sgfilt.values
        sgfilt_vals = sgfilt_vals[~np.isnan(sgfilt_vals)]
        sgfilt_001 = np.percentile(sgfilt_vals, 0.1)
        ## collect decrease more than 0.01 percentile
        sgfilt_low = df_sgfit[df_sgfit.sgfilt < sgfilt_001]
        
        """ Collect num of perturbations and dates"""
        num_pert = len(sgfilt_low)
        
        out_num_dir = out_dir + os.sep + "01_num_date"
        os.makedirs(out_num_dir,exist_ok=True)
        # sgfilt_low.to_csv(out_num_dir + os.sep + f"num_date_{filename}.txt")
    
    except: #All nan or strange dataset
        num_pert = np.nan
        
    
    num_dic[filename] = num_pert
        
    
    
    
    
    
    
    
    
    
        
 
    