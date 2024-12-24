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
# import ruptures as rpt
import math
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL #test
from scipy.fft import fft, ifft, fftfreq
from scipy import fftpack
from scipy import stats
import copy
import rasterio
from tqdm import tqdm
# os.chdir(r"C:\Users\chihiro\Desktop\Python\ANALYSIS\Resilience")
os.chdir('/Users/wtakeuchi/Desktop/Python/ANALYSIS/Resilience')
import _PSD_beta

# csvfile = "/Volumes/PortableSSD/Malaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels/A1/15706.csv"
# page = "A1"
page = sys.argv[1]
# csv_dir = rf"/Volumes/PortableSSD/MAlaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels_EVI_16days/{page}"
csv_dir = rf"/Volumes/PortableSSD/MAlaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels_EVI/{page}"
startyear = 2002
endyear = 2023
# out_dir = r"D:\Malaysia\02_Timeseries\Resilience\08_beta\_EVI16"
# out_dir = '/Volumes/SSD_2/Malaysia/02_Timeseries/Resilience/08_beta/_EVI16'
out_dir = '/Volumes/SSD_2/Malaysia/02_Timeseries/Resilience/08_beta/_EVI'

csvs = glob.glob(csv_dir + os.sep + "*.csv")

# sample_tif = rf"F:\MAlaysia\MODIS_EVI\01_MOD13A2061_resample\_4326_res01_age_adjusted\extent\MODEVI_20221016_4326_res01_adj_extentafter_{page}.tif"
sample_tif = f'/Volumes/PortableSSD/MAlaysia/MODIS_EVI/01_MOD13A2061_resample/_4326_res01_age_adjusted/extent/MODEVI_20221016_4326_res01_adj_extentafter_{page}.tif'
with rasterio.open(sample_tif) as src:
    src_arr = src.read(1)
    meta = src.meta
    transform = src.transform
    height, width = src_arr.shape[0], src_arr.shape[1]
    # profile = src.profile
    
## 参照しているラスターのAffineのheight pixelはマイナスになっていてほしい
meta.update({"nodata":np.nan})


results = {
           "psdall":{},
           "psdallresid":{},
}

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
    
    """ # extract SIF"""
    df_csv_sif = df_csv.loc[:, "EVI"]
    
    """ # z scoring""" ## -> not used
    seri_csv_sif_z = df_csv_sif.copy()
    df_csv_sif_z = seri_csv_sif_z.to_frame()
    sif_mean = df_csv_sif_z["EVI"].mean()
    sif_std = df_csv_sif_z["EVI"].std()
    df_csv_sif_z.loc[:, "EVIz"] = (df_csv_sif_z["EVI"]-sif_mean)/sif_std
    
    """ # Scaling 0-1""" ##
    seri_csv_sif_s = df_csv_sif.copy()
    df_csv_sif_s = seri_csv_sif_z.to_frame()
    sif_max = df_csv_sif_s["EVI"].max()
    sif_min = df_csv_sif_s["EVI"].min()
    df_csv_sif_s.loc[:, "EVIs"] = (df_csv_sif_s["EVI"]-sif_min)/(sif_max - sif_min)
    
    
    # data_sif = df_csv_sif_z["EVIz"]#.values #some EVI pixels not exist in early years
    data_sif = df_csv_sif_s["EVIs"]
    df_divi_valid = data_sif.interpolate(method='linear')
    df_divi_valid = df_divi_valid[~np.isnan(df_divi_valid)]
    
    nan_chck = data_sif[~np.isnan(data_sif)]
    
    if len(nan_chck) ==0: # no valid area
        for ds in results.values():
            ds[filename]=np.nan
        continue
        
    else: # there are valid values
        ## STL
        stl_result = STL(df_divi_valid, period=12).fit()
        # stl_seasonal = stl_result.seasonal
        # stl_deseason = stl_result.observed - stl_seasonal
        stl_resid = stl_result.resid
                        
        
        """
        fig = stats_result.plot()
        # fig = stl_result.plot()
        fig.set_size_inches((10, 5))
        ## if extract seasonal for plotting
        # fig, ax = plt.subplots(figsize=(10, 5))
        # stats_seasonal.plot(ax=ax)                
        fig.tight_layout()
        
        out_dir_fin = out_dir + os.sep + "_png"
        outfilename = out_dir_fin + os.sep + f"timeplot_{filename}_{ilst[0]}-{ilst[1]}.png"
        # outfilename = out_dir_fin + os.sep + f"timeplot_{filename}_{ilst[0]}-{ilst[1]}_stl.png"
        # outfilename = out_dir_fin + os.sep + f"timeplot_{filename}_{ilst[0]}-{ilst[1]}_ADFdiff.png"
        fig.savefig(outfilename)
        """
        
        
        """# PSD with original interpolated data"""
        # stats_seasonal_vali = stats_seasonal[~np.isnan(stats_seasonal)]
        df_use_psd = df_divi_valid # use original interpolated data
        try: 
           psd_beta = _PSD_beta.main(df_use_psd)
           # if psd_beta >0:
           #      psd_beta =0
        except:
            psd_beta =np.nan
        
        results["psdall"][filename] = psd_beta
        
        
        """# PSD with residual data"""
        df_use_psd = stl_resid
        try: 
           psd_beta_resid = _PSD_beta.main(df_use_psd)
           # if psd_beta >0:
           #      psd_beta =0
        except:
            psd_beta_resid =np.nan
        results["psdallresid"][filename] = psd_beta_resid
                
    

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
    
    out_file = out_dir + os.sep +f"{page}_{ky}.tif"
    with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
        with rasterio.open(out_file, 'w', **meta) as dst:
          dst.write(importance_arr_re, 1)  
        
    
    
    
