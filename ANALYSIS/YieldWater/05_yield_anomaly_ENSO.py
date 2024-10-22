# -*- coding: utf-8 -*-
"""
Find yield anomaly due to ENSO/IOD
If contain ElNino or IOD in the year
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
os.chdir("/Users/wtakeuchi/Desktop/Python/ANALYSIS/YieldWater")
import _yield_csv  
# import _csv_to_dataframe 

# enso_csv = r"F:\MAlaysia\ENSO\00_download\meiv2.csv"
# iod_csv = r"F:\MAlaysia\ENSO\10_IOD\NASA_Json.csv"
# out_dir = r"D:\Malaysia\02_Timeseries\YieldWater\04_yield_with_ENSO"
enso_csv = '/Volumes/PortableSSD/Malaysia/ENSO/00_download/meiv2.csv'
iod_csv = '/Volumes/PortableSSD/Malaysia/ENSO/10_IOD/NASA_Json.csv'
out_dir = '/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/04_yield_with_ENSO'

startdate = datetime.datetime(2002,1,1)
enddate = datetime.datetime(2023,12,1)

### Extract date when exceeding threhold
def threshold_to_date(seri,thre):## threshold
    df_valid = seri.where(seri>thre) #|(seri<thre*-1)
    df_valid = df_valid.dropna()
    valid_date = list(df_valid.index)
    # convert date to end of month
    valid_date = [ts + pd.offsets.MonthEnd(0) for ts in valid_date]
    valid_date = list(set(valid_date))
    return valid_date


""" # prepare ENSO timeseries """
enso_thre = 0.5
df_enso = pd.read_csv(enso_csv)
df_enso = df_enso.set_index("YEAR")

### # convert mei str to datetime 
mei_month_dict = {"DJ":1,"JF":2,"FM":3,"MA":4,"AM":5,"MJ":6,
                  "JJ":7,"JA":8,"AS":9,"SO":10,"ON":11,"ND":12}

df_enso_num = df_enso.rename(columns=mei_month_dict)

enso_result =[]
for yr,row in df_enso_num.iterrows():
    for m in row.index:
        datetime_ = datetime.datetime(int(yr), m, 1)
        mei_val= row.at[m]
        enso_result.append([datetime_, mei_val])
        
df_enso_series =pd.DataFrame(enso_result, columns=["datetime","enso"])
df_enso_series = df_enso_series.set_index("datetime")
## period
df_enso_series = df_enso_series.loc[startdate:enddate]
## threshold
enso_valid_date = threshold_to_date(df_enso_series, 0.5)
enso_year = list(set([y.year for y in enso_valid_date]))


""" # prepare IOD timeseries """
df_iod = pd.read_csv(iod_csv)
df_iod['date'] = pd.to_datetime(df_iod['datetime'])
df_iod = df_iod.drop("datetime",axis=1)
df_iod = df_iod.rename(columns={"date":"datetime"})
df_iod = df_iod.set_index("datetime")
## period
df_iod = df_iod.loc[startdate:enddate]
## threshold
df_iod_std = abs(df_iod).std()
df_iod_mean = abs(df_iod).mean()
iod_thre = df_iod_mean + df_iod_std
iod_valid_date = threshold_to_date(df_iod,iod_thre)
iod_year = list(set([y.year for y in iod_valid_date]))


""" yield df"""
df_yield, df_yield_z = _yield_csv.main()
nondata_region = list(df_yield_z[df_yield_z.isna().all(axis=1)].index)
# max_yield = np.nanmax(df_yield.values)
# min_yield = np.nanmin(df_yield.values)


""" yield anomaly  # use z scored yield"""
def extreme_anomaly(df, yearlist):
    # yield_enso = df_yield_regi[df_yield_regi.index.year.isin(enso_year)]
    # yield_non_enso = df_yield_regi[~df_yield_regi.index.year.isin(enso_year)]
    # yield_enso_mean = yield_enso.mean().values[0]
    # yield_nonenso_mean = yield_non_enso.mean().values[0]
    # ### anomaly : enso - normal
    # yield_enso_anomaly =yield_enso_mean - yield_nonenso_mean
    df_extreme = df[df.index.year.isin(yearlist)]
    df_nonextreme = df[~df.index.year.isin(yearlist)]
    extreme_mean = df_extreme.mean().values[0]
    nonextreme_mean = df_nonextreme.mean().values[0]
    ### anomaly : enso - normal
    extreme_anomaly =extreme_mean - nonextreme_mean
    
    return extreme_anomaly
    

enso_ano_result = {}
iod_ano_result = {}
for regi, row in tqdm(df_yield_z.iterrows()):
    yield_result = []
    for yer in row.index:
        dt_yield = datetime.datetime(int(yer),12,31)
        yield_val = row[yer]
        yield_result.append([dt_yield, yield_val])
        
    df_yield_regi = pd.DataFrame(yield_result, columns=["datetime","yield"])
    df_yield_regi = df_yield_regi.set_index("datetime")
    df_yield_regi = df_yield_regi.dropna()
    
    ### anomaly : enso - normal
    yield_enso_anomaly = extreme_anomaly(df_yield_regi, enso_year)
    yield_iod_anomaly = extreme_anomaly(df_yield_regi, iod_year)
    
    enso_ano_result[regi] = [yield_enso_anomaly]
    iod_ano_result[regi] = [yield_iod_anomaly]
    
    
df_enso_result = pd.DataFrame.from_dict(enso_ano_result).T
df_enso_result = df_enso_result.rename(columns={0:"anomalyENSO"})

df_iod_result = pd.DataFrame.from_dict(iod_ano_result).T
df_iod_result = df_iod_result.rename(columns={0:"anomalyIOD"})

df_concat = pd.concat([df_enso_result, df_iod_result],axis=1)
df_concat.to_csv(out_dir+os.sep+"ENSOIOD_yield_anomaly.csv")

                
            
            
        
        


