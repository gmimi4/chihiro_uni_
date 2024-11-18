# -*- coding: utf-8 -*-
"""
numpy: https://labo-code.com/python/wavelet-coherence-pahse/
Using this: Pywavelet: https://atatat.hatenablog.com/entry/data_proc_python10
PyWavelt: https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
Scipy: https://qiita.com/yukiB/items/59f8484e72bb0471ad47
"""

import numpy as np
import pandas as pd
import os, sys
import glob
import time
import pywt
from tqdm import tqdm
import rasterio
import matplotlib.pyplot as plt  # for display purposes
import matplotlib.dates as mdates

import matplotlib as mpl


import math
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL #test
from scipy.fft import fft, ifft, fftfreq
from scipy import fftpack
from scipy import stats
from scipy import signal


# csvfile = "/Volumes/PortableSSD/Malaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels/A1/15706.csv"
page = "A1"
page = sys.argv[1]
csv_dir = rf"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels_until2023\{page}"
startyear = 2002
endyear = 2023
out_dir = r"D:\Malaysia\02_Timeseries\Wavelet"

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

varlist = ['GOSIF','rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD']



for csvfile in tqdm(csvs):
    # 7460 error
    # csvfile = r"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels_until2023\A1\7443.csv" #5240 no changep #7443
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
    
    
    """ # Scaling 0-1""" ##
    for var in varlist:
        df_var = df_csv.loc[:,var]
        df_var_nan = df_var.dropna()
        var_max = df_var.max()
        var_min = df_var.min()
        df_csv.loc[:, f"{var}s"] = (df_var-var_min)/(var_max - var_min)
        num_year = len(set(list(df_csv.index.year)))
    
    
    """ Run """
    # data_sif = df_csv_sif_z["GOSIFz"]#.values #some GOSIF pixels not exist in early years    
    for var in varlist:
        s_var = df_csv.loc[:,f"{var}s"]
        scales = np.arange(1, 12*2)
        cwtmatr,freq = pywt.cwt(s_var, scales, 'morl')
        # plt.imshow(cwtmatr,aspect='auto') #fft
        fig, ax = plt.subplots(figsize=(10,5))
        im = ax.imshow(cwtmatr, aspect='auto', cmap = "rainbow",
            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        # plt.colorbar()
        jan_dates = pd.date_range(start=f"{startyear}-01-01", end=f"{endyear}-01-01", freq="AS")
        tick_positions = np.linspace(0, len(df_csv) - 1, len(jan_dates))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(jan_dates.strftime('%Y-%m-%d'), rotation=45, ha="right")
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(out_dir + os.sep + f"cwt_{var}_{filename}.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        
       
        
    
    
    
