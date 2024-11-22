# -*- coding: utf-8 -*-
"""
# compute months to recover from El ninp damage
# sum up months
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
from dateutil.relativedelta import relativedelta
from statistics import mean

pagename = "A1"
mei_csv = r"F:\MAlaysia\ENSO\00_download\meiv2.csv"
in_dir_parent = r"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels_until2023"
out_dir_parent = r"D:\Malaysia\02_Timeseries\Resilience\04_recovery"
timerange = 0 #months: period that elnino impact may last 今は適当
out_dir_parent = out_dir_parent + os.sep + f"timerange{timerange}"
os.makedirs(out_dir_parent,exist_ok=True)

sample_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_01"
# pagename_list = ["A1", "A2", "A3", "A4"]

startyear = 2002
endyear = 2023
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

# """ # targeting to super Elnino from 2015-2016"""
# elnino_list_sup = [e for e in elnino_list if ((e[0] ==2015) or(e[0] ==2016)) ]


""" # convert mei str to datetime """
### mei string dic ###
# mei_month_dict = {"DJ":[12,1],"JF":[1,2],"FM":[2,3],"MA":[3,4],"AM":[4,5],"MJ":[5,6],
#                   "JJ":[6,7],"JA":[7,8],"AS":[8,9],"SO":[9,10],"ON":[10,11],"ND":[11,12]}
mei_month_dict = {"DJ":1,"JF":2,"FM":3,"MA":4,"AM":5,"MJ":6,
                  "JJ":7,"JA":8,"AS":9,"SO":10,"ON":11,"ND":12}

### obtain target month from mei string
def convert_mei_date(mei_list):
    mei_list_date = []
    for me in mei_list:
        yearint = int(me[0])
        mei_str = me[1]        
        month1 = mei_month_dict[mei_str]            
        mei_list_date.append([yearint, month1]) #use later month # otherwise overlap
    
    mei_list_date = sorted(mei_list_date, key=lambda x: (x[0], x[1]))
    
    return mei_list_date
											    

### convert to datetime ###
elnino_list_date = convert_mei_date(elnino_list)
# elnino_list_sup_date = convert_mei_date(elnino_list_sup)
lanina_list_date = convert_mei_date(lanina_list)
neutral_list_date = convert_mei_date(neutral_list)

### 同じ年のは同じelninoとしてまとめる（始まりの月のみ取得）
yr_list = list(set([y[0] for y in elnino_list_date]))

ym_dict = {}
for yr in yr_list:
    month_list = []
    for ym in elnino_list_date:
        if ym[0] == yr:
            month_list.append(ym[1])
    ym_dict[yr] = month_list
            
elnino_date =[]
for yr, mlist in ym_dict.items():
    mindate = min(mlist)
    maxdate = max(mlist)
    elnino_date.append([yr,mindate,maxdate])
    
elnino_date = sorted(elnino_date, key=lambda x: x[0])
### start from 2003
elnino_date = elnino_date[1:]


months = [m for m in range(1,13)]

""" # dfからMonthly mean dicを得る"""
def get_monthly_mean_dic(df):
    #df=df_seasonal
    month_dic={}
    for m in months:
        month_row = df[df.index.month == m]
        monthly_mean = month_row.mean()
        month_dic[m] = monthly_mean
    
    return month_dic


""" #Process """

in_dir = os.path.join(in_dir_parent, pagename)
csv_list = glob.glob(in_dir + os.sep + "*.csv")

recovery_result = {}

for csvfile in tqdm(csv_list):
    # csvfile = [c for c in csv_list if "15706" in c][0]
    filename = os.path.basename(csvfile)[:-4]
    df_csv = pd.read_csv(csvfile, index_col ='datetime', parse_dates=['datetime']) 
    
    """ # set period"""
    df_csv = df_csv[((df_csv.index.year >= startyear) & (df_csv.index.year <= endyear)) ]

    """ # extract SIF"""
    df_csv_sif = df_csv.loc[:, "GOSIF"]
    
    """ #deseasonal""" 
    # obtain monthly mean for each month
    monthly_mean_org = get_monthly_mean_dic(df_csv_sif)
    
    df_csv_sif_de = df_csv_sif.to_frame()
        
    for m in months:
        specific_month_rows = df_csv_sif[df_csv_sif.index.month == m]
        monthly_mean = monthly_mean_org[m]
        specific_month_idx = specific_month_rows.index.tolist()
        df_csv_sif_de.loc[specific_month_idx, f"GOSIFde"] = specific_month_rows-monthly_mean
    
    ### set df_csv_sif again
    df_csv_sif = df_csv_sif_de.GOSIFde
    
    
    ## check nan
    df_csv_sif = df_csv_sif[~np.isnan(df_csv_sif)]
    if len(df_csv_sif) ==0:
        recovery_result[int(filename)] = np.nan
        continue
    else:
        """ # get 1 year average before elnino"""
        recovery_months = []
        for i,ym in enumerate(elnino_date):
    
            ### one year average before elnino ###
            datetime_start = pd.to_datetime(f'{ym[0]}-{ym[1]:02d}')
            datetime_start_yrbefore = datetime_start - pd.DateOffset(years=1)
            df_before =  df_csv_sif[(df_csv_sif.index >datetime_start_yrbefore)&(df_csv_sif.index <datetime_start)]
            ## target val to recover
            df_before_ave = df_before.mean()
            
            """ # minimum val during elnino """
            ## elnino end date
            try:
                datetime_end = pd.to_datetime(f'{ym[0]}-{ym[2]+timerange:02d}')
            except ValueError:
                datetime_end = pd.to_datetime(f'{ym[0]+1}-{ym[2]+timerange-12:02d}')
            ## elnino min val date
            df_during =  df_csv_sif[(df_csv_sif.index >datetime_start)&(df_csv_sif.index <datetime_end)]
            if len(df_during) >0: #some pixels have data after 2019
                date_min = df_during.idxmin()
            else:
                continue
            
            """ # recovery time after elnino """
            ## before next elnino
            if i != len(elnino_date)-1: #if not last i
                ym_next = elnino_date[i+1]
                datetime_next = pd.to_datetime(f'{ym_next[0]}-{ym_next[1]:02d}') #next elnino start
            else: #last i
                datetime_next = pd.to_datetime('2023-12')
            df_after =  df_csv_sif[(df_csv_sif.index >date_min)&(df_csv_sif.index <datetime_next)]
            if len(df_after)>0:
                try:
                    date_recover = df_after[df_after>=df_before_ave].index[0]
                    period_recover = relativedelta(date_recover,date_min).months
                except IndexError: #cannot recover
                    period_recover = relativedelta(datetime_next,date_min).months
            else:#ゆっくり減った功績も考慮する
                period_recover = relativedelta(datetime_next,date_min).months
                
            recovery_months.append(period_recover)
            ## sumup ###
            # recovery_months_sum = sum(recovery_months)
            ## mean ###  #because some pixels start from 2019
            recovery_months_sum = mean(recovery_months)

            
    recovery_result[int(filename)] = recovery_months_sum

    

""" #export to tif """
# sample tif
sample_tif = sample_dir + os.sep + f"{pagename}_p_values_importance_2002-2022.tif"
with rasterio.open(sample_tif) as src: # pval tif as sample tif
    arr = src.read(1)
    profile=src.profile
    height, width = arr.shape[0],arr.shape[1]
    

## process for each dic
#念のためsort by filename (idx name)
all_resid_sort = sorted(recovery_result.items())

var_cv_arr = np.array([c[1] for c in all_resid_sort]) #順番通りに取り出しているはず 
var_cv_reshape = var_cv_arr.reshape((height, width))


out_dir =  out_dir_parent + os.sep + "mean_time"
os.makedirs(out_dir, exist_ok=True)

outfile = os.path.join(out_dir,f"{pagename}_recovery_meantime.tif")
with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
    with rasterio.open(outfile, "w", **profile) as dst:
        dst.write(var_cv_reshape, 1)
                
        
 
    