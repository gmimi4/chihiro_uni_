# -*- coding: utf-8 -*-
"""
Extract global coherence (mean) and its change time
Precise COI is needed. At present, longer than 6 month freq is used.
"""

import pandas as pd
import os, sys
import glob
import numpy as np
import time
# import pywt
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


in_dir_parent = r"D:\Malaysia\02_Timeseries\Wavelet\01_cross_correlation"

reginames = [r for r in os.listdir(in_dir_parent) if os.path.isdir(os.path.join(in_dir_parent, r))]

varlist = ['rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD'] 

scale_upper = 13
for regi in tqdm(reginames):
    in_dir = in_dir_parent + os.sep + regi
    coh_files = glob.glob(in_dir + os.sep + "coherence_*.npy")
    coh_list = []
    for var in varlist:
        coh_file = [c for c in coh_files if var in c][0]
        arr_coh = np.load(coh_file)
        arr_mean = np.mean(arr_coh,axis=1) # mena for row direction
        freq_file = in_dir + os.sep + f"frequency_{var}.npy"
        arr_freq = np.load(freq_file)
        
        """ Select from scale range"""
        scale_file = glob.glob(in_dir + os.sep + f"scale_{var}.npy")[0]
        arr_scale = np.load(scale_file)

        ### Plot 
        ## select f shortre than upper scale
        arr_freq_sel = arr_freq[arr_scale < scale_upper]
        
        fig = plt.figure(figsize=(5,5))
        plt.plot(arr_mean,  arr_freq)
        plt.axhline(y=arr_freq_sel[-1], color='grey', linestyle='--', linewidth=0.7, label=f'scale:12 month')
        plt.ylabel("frequency")
        plt.xlabel("coherence mean")
        plt.legend()
        fig.savefig(in_dir + os.sep + f"coherence_mean_{var}.png")
        plt.close()
        
        ## select coherence shortre than upper scale
        arr_mean_sel = arr_mean[arr_scale < scale_upper]
        ## Highest coherence scale
        scale_high = np.argmax(arr_mean_sel)
        
        """ Extract coherence at this scale and mean by years"""
        arr_coh_sel = arr_coh[scale_high]
        date_file = in_dir + os.sep + f"datetime_{var}.npy"
        arr_date = np.load(date_file)
        ### Convert to dataframe
        df_coh = pd.DataFrame({f'{var}coh': arr_coh_sel}, index=arr_date)
        df_coh_year = df_coh.resample("Y").mean()
        coh_list.append(df_coh_year)
    
    df_coh_concat = pd.concat(coh_list, axis=1)
    out_dir = in_dir
    df_coh_concat.to_csv(out_dir + os.sep + f"{regi}_coh_change.csv" )
        
        
   
