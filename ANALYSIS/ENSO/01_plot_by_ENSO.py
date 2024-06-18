# -*- coding: utf-8 -*-
"""
#1. Identify ElNino, LaNiNa, Neutral by MEI.v2 of NOAA
#2. obtain deviation from average for nino, nina, neutral
#3. considering time lag
"""

import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime
import calendar
import itertools
from tqdm import tqdm
import rasterio

mei_csv = r"F:\MAlaysia\ENSO\00_download\meiv2.csv"
in_dir_parent = r"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels"
out_dir_parent = r"F:\MAlaysia\ENSO\01_deviations"
sample_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_01"
# pagename_list = ["A1", "A2", "A3", "A4"]
pagename = "A1"

startyear = 2002
endyear = 2022
k_th = 0 #time lag

months = [m+1 for m in range(12)]

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


elnino_list = [e for e in elnino_list if (int(e[0])>=startyear)&(int(e[0])<=endyear)]
lanina_list = [e for e in lanina_list if (int(e[0])>=startyear)&(int(e[0])<=endyear)]
neutral_list = [e for e in neutral_list if (int(e[0])>=startyear)&(int(e[0])<=endyear)]


### mei string dic ###
mei_month_dict = {"DJ":[12,1],"JF":[1,2],"FM":[2,3],"MA":[3,4],"AM":[4,5],"MJ":[5,6],
                  "JJ":[6,7],"JA":[7,8],"AS":[8,9],"SO":[9,10],"ON":[10,11],"ND":[11,12]}

### obtain target month from mei string
def convert_mei_date(mei_list, ninonina_neut):
    mei_list_date = []
    for me in mei_list:
        yearint = int(me[0])
        mei_str = me[1]
        
        ## if elnino or lanina, time lagged rows are extracted
        if ninonina_neut == "elnino" or ninonina_neut =="lanina":
            month1 = mei_month_dict[mei_str][0] +k_th
            month2 = mei_month_dict[mei_str][1] +k_th
        ## for neutral period, no time lag is considered
        else:
            month1 = mei_month_dict[mei_str][0]
            month2 = mei_month_dict[mei_str][1]
            
        lastday1 = calendar.monthrange(yearint, month1)
        lastday2 = calendar.monthrange(yearint, month2)
        yyyymmdd_1 =  datetime(yearint, month1, lastday1[1])
        yyyymmdd_2 =  datetime(yearint, month2, lastday2[1])
        
        mei_list_date.append([yyyymmdd_1]) #, yyyymmdd_2 # adopt first month, otherwise overlap
    
    mei_list_date = list((itertools.chain.from_iterable(mei_list_date)))
    mei_list_date = sorted(list(set(mei_list_date)))
    
    return mei_list_date
											    

### convert to datetime ###
elnino_list_date = convert_mei_date(elnino_list, "elnino")
lanina_list_date = convert_mei_date(lanina_list, "lanina")
neutral_list_date = convert_mei_date(neutral_list, "neutral")

# len(elnino_list_date), len(lanina_list_date), len(neutral_list_date)


""" # define to get deviation between monthyl mean"""

def get_deviation(df_mei_date, mean_dic):
    # df_mei_date=df_csv_elnino
    # mean_dic = monthly_mean_std_dic
    vars_list = df_mei_date.columns.tolist()
    
    monthly_devi_dic = {}
    for var in vars_list:
        df_var = df_mei_date.loc[:,var]
        month_devi = {}
        for m in months:
            specific_month_rows = df_var[df_var.index.month == m]
            monthly_mean = mean_dic[var][m]
            devis = specific_month_rows - monthly_mean # mixture of + and -
            devis_sum = sum(devis)
            month_devi[m] = devis_sum
                
        monthly_devi_dic[var] = month_devi #確認用
    
    return monthly_devi_dic
    
    


""" #Process """

in_dir = os.path.join(in_dir_parent, pagename)
csv_list = glob.glob(in_dir + os.sep + "*.csv")

elnino_result = {}
lanina_result = {}
neutral_result = {}

for csvfile in tqdm(csv_list):
    # csvfile = [c for c in csv_list if "15706" in c][0]
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

    
    """ # calculate monthyl mean"""
    vars_list = df_csv.columns.tolist()
    
    ### obtain monthly mean and std as dict
    monthly_mean_std_dic = {}
    for var in vars_list:
        df_var = df_csv.loc[:,var]
        month_dic = {}
        for m in months:
            specific_month_rows = df_var[df_var.index.month == m]
            monthly_mean = specific_month_rows.mean(skipna=True)
            # monthly_std = specific_month_rows.std(skipna=True)
            month_dic[m] = monthly_mean
                
        monthly_mean_std_dic[var] = month_dic #確認用
    
    
    """ # obtain deviation of nino, nina, neut from monthyl mean"""
    df_csv_elnino = df_csv.loc[elnino_list_date]
    df_csv_lanina = df_csv.loc[lanina_list_date]
    df_csv_neutral = df_csv.loc[neutral_list_date]
    
    ## this is deviation sum dictionary by var
    elnino_devi_dic = get_deviation(df_csv_elnino, monthly_mean_std_dic) #monthyl deviation sum by var, dictionary
    lanina_devi_dic = get_deviation(df_csv_lanina, monthly_mean_std_dic)
    neutral_devi_dic = get_deviation(df_csv_neutral, monthly_mean_std_dic)
    
    elnino_sum ={}
    lanina_sum ={}
    neutral_sum ={}
    for var in vars_list:
        elnino_dev_sum_var = sum(elnino_devi_dic[var].values())
        lanina_dev_sum_var = sum(lanina_devi_dic[var].values())
        neutral_dev_sum_var = sum(neutral_devi_dic[var].values())
        
        elnino_sum[var] = elnino_dev_sum_var
        lanina_sum[var] = lanina_dev_sum_var
        neutral_sum[var] = neutral_dev_sum_var
        
    elnino_result[int(filename)] = elnino_sum
    lanina_result[int(filename)] = lanina_sum
    neutral_result[int(filename)] = neutral_sum
    


""" #export to tif """
out_dir =  out_dir_parent + os.sep + "lag"+ str(k_th)
os.makedirs(out_dir, exist_ok=True)

# sample tif
sample_tif = sample_dir + os.sep + f"{pagename}_p_values_importance_2002-2022.tif"
with rasterio.open(sample_tif) as src: # pval tif as sample tif
    arr = src.read(1)
    profile=src.profile
    height, width = arr.shape[0],arr.shape[1]
    

## process for each dic
for cv_files in [elnino_result, lanina_result, neutral_result]:
    #念のためsort by filename (idx name)
    all_resid_sort = sorted(cv_files.items())
    
    for variable in  vars_list: 
        var_cv_arr = np.array([c[1][variable] for c in all_resid_sort]) #順番通りに取り出しているはず 
        var_cv_reshape = var_cv_arr.reshape((height, width))
    
        outfile = os.path.join(out_dir,f"{pagename}_deviation_{variable}_{startyear}-{endyear}_lag{k_th}.tif")
        with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
            with rasterio.open(outfile, "w", **profile) as dst:
                dst.write(var_cv_reshape, 1)
            
        
 
    