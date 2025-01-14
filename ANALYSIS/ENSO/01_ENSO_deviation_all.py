# -*- coding: utf-8 -*-
"""
#1. Identify ElNino by MEI.v2. From month over 0.5 to 12 months.
#2. obtain deviation in average for all El Nino event
#2-2. Divide period by 3 months 
#3. considering time lag
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

mei_csv = r"F:\MAlaysia\ENSO\00_download\meiv2.csv"
in_dir_parent = r"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels_EVI"
out_dir = r"F:\MAlaysia\ENSO\01_deviations\_allevents"
# pagename = "A1"
pagename = sys.argv[1]

# sample tif
sample_tif = rf"F:\MAlaysia\MODIS_EVI\01_MOD13A2061_resample\_4326_res01_age_adjusted\extent\MODEVI_20221016_4326_res01_adj_extentafterFIN_{pagename}.tif"
with rasterio.open(sample_tif) as src: # pval tif as sample tif
    arr = src.read(1)
    profile=src.profile
    height, width = arr.shape[0],arr.shape[1]

startyear = 2002
endyear = 2023
k_th = 0 #time lag, not used so far

months = [m+1 for m in range(12)]

mei_month_dict = {"DJ":1,"JF":2,"FM":3,"MA":4,"AM":5,"MJ":6,
                  "JJ":7,"JA":8,"AS":9,"SO":10,"ON":11,"ND":12}
season_dic = {"DJF":[12,1,2], "MAM":[3,4,5],"JJA":[6,7,8],"SON":[9,10,11]}
season_mons = list(season_dic.values())

""" # Identify months of ElNino, LaNiNa, Neutral"""
enso_thre = 0.5

df_mei = pd.read_csv(mei_csv)
df_mei = df_mei.set_index("YEAR")

elnino_list, lanina_list, neutral_list = [],[],[]

for year, row in df_mei.iterrows():
    for colname, value in row.items():
        if value >enso_thre:
            elnino_list.append([year, colname])
        elif value < -1 * enso_thre:
            lanina_list.append([year, colname])
        else:
            neutral_list.append([year, colname])

## filtering period
elnino_list = [e for e in elnino_list if (int(e[0])>=startyear)&(int(e[0])<=endyear)]
lanina_list = [e for e in lanina_list if (int(e[0])>=startyear)&(int(e[0])<=endyear)]
neutral_list = [e for e in neutral_list if (int(e[0])>=startyear)&(int(e[0])<=endyear)]


""" # convert mei str to datetime """
def convert_mei_date(mei_list):
    mei_list_date = []
    for me in mei_list:
        yearint = int(me[0])
        mei_mon = mei_month_dict[me[1]]
        lastday1 = calendar.monthrange(yearint, mei_mon) #num of weeks, days
        yyyymmdd_1 =  datetime(yearint, mei_mon, lastday1[1])
        mei_list_date.append([yyyymmdd_1])
    
    mei_list_date = list((itertools.chain.from_iterable(mei_list_date)))
    mei_list_date = sorted(list(set(mei_list_date)))
    
    return mei_list_date
											    

### convert to datetime ###
elnino_list_date = convert_mei_date(elnino_list)
lanina_list_date = convert_mei_date(lanina_list) # not used
neutral_list_date = convert_mei_date(neutral_list) # not used

### ElNino from to #start from 0.5< till 1 year later
elnino_years = list(set([y.year for y in elnino_list_date]))
elnino_from_to =  []
for yr in elnino_years:
    elenino_months = [e for e in elnino_list_date if e.year==yr ]
    elenino_earliest = min(elenino_months)
    elenino_end = elenino_earliest + relativedelta(months=11)
    elnino_from_to.append([elenino_earliest, elenino_end])



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

    """ # calculate monthyl mean and anomly in whole period"""
    df_csv_use = df_csv.copy()
    vars_list = df_csv_use.columns.tolist()
    
    # monthly_mean_std_dic = {}
    for var in vars_list:
        df_var = df_csv_use.loc[:,var]
        df_csv_use[f"{var}ano"] = np.nan
        # month_dic = {}
        for m in months:
            specific_month_rows = df_var[df_var.index.month == m]
            monthly_mean = specific_month_rows.mean(skipna=True)
            df_csv_use.loc[specific_month_rows.index, f"{var}ano"] = specific_month_rows - monthly_mean
            # month_dic[m] = monthly_mean
        # monthly_mean_std_dic[var] = month_dic
        
        
    """ # obtain deviation by 3 months periodsduring nino"""    
    devi_dic = {}
    devi_mon_dic = {}
    for var in vars_list:
        df_var = df_csv_use.loc[:,f"{var}ano"]     
        
        month_devi = {} #input monthl series for var
        for s_e in elnino_from_to:
            start = s_e[0]
            end = s_e[1]
            df_var_nino = df_var[(df_var.index > start)&(df_var.index < end)]
            for ms in season_mons:
                df_var_nino_mon = df_var_nino[df_var_nino.index.month.isin(ms)]
                ano_mean = df_var_nino_mon.mean()
                month_devi[ms[0]] = ano_mean
        
        ano_min = min(month_devi.values())
        month_min = min(month_devi, key=month_devi.get)
        
        devi_dic[var] = ano_min
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

    outfile = os.path.join(out_dir,f"{pagename}_mindevi_{variable}.tif")
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
        
 
    