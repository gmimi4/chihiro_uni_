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
from tqdm import tqdm


pagename = sys.argv[1]
# pagename = 'A1'
in_dir = f'/Volumes/PortableSSD/MAlaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels_EVI_16days/{pagename}'
out_dir_parent = '/Volumes/SSD_2/Malaysia/02_Timeseries/Resilience/07_perturbation'
out_dir = out_dir_parent + os.sep + f"{pagename}"
os.makedirs(out_dir,exist_ok=True)

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
timing_dic = {}
recovery_dic ={}
recovrate_dic = {}
r2_dic = {}
pertutbtions = [] 
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
        
        """ Collect num of perturbations and timing"""
        ## num
        num_pert = len(sgfilt_low)
        ## timing (assuming 1 point)
        timing = series_reset[series_reset.datetime==sgfilt_low.index.values[0]].index[0]
        
        ## collect (later) perturbation df
        sgfilt_low_reset = sgfilt_low.reset_index()
        sgfilt_low_reset["filename"] = filename
        
        
        # -------------------------
        """ # Local minimum of resid within 2 month"""
        # -------------------------
        ## assume 1 point perturbation
        df_sgfit_min = df_sgfit.loc[df_sgfit['sgfilt'].idxmin()]
        perturbation_date = df_sgfit_min.name
        perturbation_date_2mon = perturbation_date + pd.DateOffset(months=2)
        ## local minumum date
        ser_resid_2mon = ser_resid.loc[perturbation_date:perturbation_date_2mon]
        perturbation_least = ser_resid_2mon.idxmin() # This is start of recovery
        
        
        # -------------------------
        """ # Fitting for recovery"""
        # -------------------------
        def recovery_exponential(t, x0, r): #variable should be first
            return x0 * np.exp(r * t)
        
        ## まずは5yrでやってみる
        fit_period = 5
        dataset = ser_resid.loc[perturbation_least: perturbation_least + pd.DateOffset(years=fit_period)]
        dataset = dataset.reset_index()
        x_data = dataset.index
        y_data = dataset.resid
        
        """ #following paper code """
        # dataset_raw = ser_fit.observed
        # dataset_raw = dataset_raw.loc[perturbation_least: perturbation_least + pd.DateOffset(years=fit_period)]
        # fitting = dataset.set_index("datetime")
        # armin = np.argmin(fitting.values[:8])
        # fitting_min = fitting[armin:]
        # trange = np.arange(fitting_min.shape[0])
        # popt, _ = curve_fit(recovery_exponential, trange, fitting_min.values - np.nanmean(fitting_min.values), p0=p0, jac=exp_jac, bounds=bounds)
        # test = fitting_min.values - np.nanmean(fitting_min.values)
        # plt.scatter(range(len(test)), test, color='red', label="1D Array Points")
        # plt.show()
        """ """
        
        ## initial guess
        x0_init = ser_resid.loc[perturbation_least - pd.DateOffset(years=fit_period): perturbation_least - pd.DateOffset(years=1)]
        x0_init = x0_init.mean(skipna=True)
        initial_guess = [x0_init, -0.1]  # Initial guess for [x0, r]
        ## fit
        params, covariance = curve_fit(recovery_exponential, x_data, y_data - np.nanmean(y_data.values), p0=initial_guess) #center to zero

        ## Generate fitted curve
        y_pred = recovery_exponential(x_data, *params)
        recovrate = params[1]
        ## Plot
        # plt.scatter(x_data, y_data, label="residual", color="blue", alpha=0.6)
        # plt.plot(x_data, y_fit, label="Fitted Curve", color="red", linewidth=2)
        # # plt.xlabel("x")
        # plt.ylabel("residual")
        # plt.legend()
        # plt.show()
        
        ## R2
        r_squared = r2_score(y_data, y_pred)
        
        # if r_squared >0.2:
        #     recovery_time = 1/np.abs(recovrate) #16days per unit
        # else:
        #     recovery_time = np.nan
        recovery_time = 1/np.abs(recovrate) #16days per unit
        
    
    except: #All nan or strange dataset
        num_pert = np.nan
        timing = np.nan
        recovery_time = np.nan
        sgfilt_low_reset = pd.DataFrame({"datetime":[np.nan], "resid":[np.nan], "sgfilt":[np.nan], "filename":[filename]})
        recovrate = np.nan
        r_squared = np.nan
        

        
    
    num_dic[filename] = num_pert
    timing_dic[filename] = timing
    recovery_dic[filename] = recovery_time
    pertutbtions.append(sgfilt_low_reset)
    recovrate_dic[filename] = recovrate
    r2_dic[filename] = r_squared



""" Export perturbation evernts"""
df_perturbation_timings = pd.concat(pertutbtions)
df_perturbation_timings.to_csv(out_dir + os.sep + f"num_date_{pagename}.txt")
    
    
    
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


to_raster_dic = {"numperturbation":num_dic, 
                 "timing":timing_dic,
                 "recoverytime":recovery_dic,
                 "recoveryrate":recovrate_dic,
                 "r2":r2_dic
                 }


for outrasname, tar_dic in to_raster_dic.items():
    to_raster(tar_dic, outrasname)
        
    
    
    
    
    
    
    
    
    
        
 
    