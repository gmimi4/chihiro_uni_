# -*- coding: utf-8 -*-
"""
# find min of 3-month average season in year
"""

import numpy as np
import pandas as pd
import os,sys
import glob
from datetime import datetime
from dateutil.relativedelta import relativedelta
import calendar
import itertools
from tqdm import tqdm
import rasterio


# in_dir_parent = r"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels_EVI"
# out_dir = 
in_dir_parent = '/Volumes/PortableSSD/Malaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels_EVI'
out_dir = '/Volumes/PortableSSD/Malaysia/ENSO/04_mins'

# pagename = "A1"
pagename = sys.argv[1]

# sample tif
# sample_tif = rf"F:\MAlaysia\MODIS_EVI\01_MOD13A2061_resample\_4326_res01_age_adjusted\extent\MODEVI_20221016_4326_res01_adj_extentafterFIN_{pagename}.tif"
sample_tif = f'/Volumes/PortableSSD/Malaysia/MODIS_EVI/01_MOD13A2061_resample/_4326_res01_age_adjusted/extent/MODEVI_20221016_4326_res01_adj_extentafterFIN_{pagename}.tif'
with rasterio.open(sample_tif) as src: # pval tif as sample tif
    arr = src.read(1)
    profile=src.profile
    height, width = arr.shape[0],arr.shape[1]

startyear = 2002
endyear = 2023

vars_list =['EVI', 'rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD']

months = [m+1 for m in range(12)]

mei_month_dict = {"DJ":1,"JF":2,"FM":3,"MA":4,"AM":5,"MJ":6,
                  "JJ":7,"JA":8,"AS":9,"SO":10,"ON":11,"ND":12}
season_dic = {"DJF":[12,1,2], "MAM":[3,4,5],"JJA":[6,7,8],"SON":[9,10,11]}
season_mons = list(season_dic.values())


""" #Process """
in_dir = os.path.join(in_dir_parent, pagename)
csv_list = glob.glob(in_dir + os.sep + "*.csv")

elnino_result = {}
elnino_mon_result ={}
for csvfile in tqdm(csv_list):
    # csvfile = [c for c in csv_list if "13820" in c][0]
    filename = os.path.basename(csvfile)[:-4]
    df_csv = pd.read_csv(csvfile, index_col ='datetime', parse_dates=['datetime'])

    """ # AMSREとAMSR2のギャップを補正する"""   
    pi_e = df_csv.loc[:,["SMDSCE","VODDSCE"]]
    pi_2 = df_csv.loc[:,["SMDSC2","VODDSC2"]]
    # calc median
    pi_e_med = pi_e.median()
    pi_2_med = pi_2.median()
    ratio_sm = pi_2_med.SMDSC2 /pi_e_med.SMDSCE
    ratio_vod = pi_2_med.VODDSC2 /pi_e_med.VODDSCE
    df_csv["SMDSCErev"] = df_csv["SMDSCE"] *ratio_sm
    df_csv["VODDSCErev"] = df_csv["VODDSCE"] *ratio_vod
    
    
    """ #SMとVODは平均にする"""
    df_csv["SM"] = df_csv[["SMDSCErev", "SMDSC2"]].mean(skipna=True, axis='columns')
    df_csv["VOD"] = df_csv[["VODDSCErev", "VODDSC2"]].mean(skipna=True, axis='columns')
    df_csv= df_csv.drop(["SMDSCE","SMDSC2","VODDSCE","VODDSC2", "SMDSCErev", "VODDSCErev"], axis=1)
    
    
    """ # set period"""
    df_csv = df_csv[((df_csv.index.year >= startyear) & (df_csv.index.year <= endyear)) ]
        
        
    """ # obtain min of 3 month average """    
    devi_dic = {}
    devi_mon_dic = {}
    for var in vars_list:
        df_var = df_csv.loc[:,f"{var}"] 
        month_ave={}
        for ms in season_mons:
            df_var_season = df_var[df_var.index.month.isin(ms)]
            season_mean = df_var_season.mean()
            month_ave[ms[0]] = season_mean
        
        ave_min = min(month_ave.values())
        month_min = min(month_ave, key=month_ave.get)
        
        devi_dic[var] = ave_min
        devi_mon_dic[var] = month_min
    

    elnino_result[int(filename)] = devi_dic
    elnino_mon_result[int(filename)] = devi_mon_dic

    

""" #export to tif """
## process for each dic
#念のためsort by filename (idx name)
all_resid_sort = sorted(elnino_result.items())

for variable in vars_list:
    var_cv_arr = np.array([c[1][variable] for c in all_resid_sort]) #順番通りに取り出しているはず 
    var_cv_reshape = var_cv_arr.reshape((height, width))

    outfile = os.path.join(out_dir,f"{pagename}_min_{variable}.tif")
    with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
        with rasterio.open(outfile, "w", **profile) as dst:
            dst.write(var_cv_reshape, 1)


all_resid_sort = sorted(elnino_mon_result.items())
for variable in vars_list:
    var_cv_arr = np.array([c[1][variable] for c in all_resid_sort]) #順番通りに取り出しているはず 
    var_cv_reshape = var_cv_arr.reshape((height, width))

    outfile = os.path.join(out_dir,f"{pagename}_minmon_{variable}.tif")
    with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
        with rasterio.open(outfile, "w", **profile) as dst:
            dst.write(var_cv_reshape, 1)              
        
 
    