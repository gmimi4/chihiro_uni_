# -*- coding: utf-8 -*-
"""
Plot of yields with ENSO and IOD
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import calendar
import datetime
import itertools
import matplotlib.pyplot as plt
os.chdir(r"C:\Users\chihiro\Desktop\Python\ANALYSIS\YieldWater")
import _yield_csv  
# import _csv_to_dataframe 

enso_csv = r"F:\MAlaysia\ENSO\00_download\meiv2.csv"
iod_csv = r"F:\MAlaysia\ENSO\10_IOD\NASA_Json.csv"
out_dir = r"D:\Malaysia\02_Timeseries\YieldWater\04_yield_with_ENSO"

# varlist = ['GOSIF', 'rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD']
# units = {'GOSIF':"W/m2/μm/sr/month", 'rain':"mm", 'temp':"degreeC", 
#          'VPD':"hPa", 'Et':"mm/day", 'Eb':"mm/day", 'SM':"m3/m3", 'VOD':""}

startdate = datetime.datetime(2002,1,1)
enddate = datetime.datetime(2023,12,1)

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


""" # prepare IOD timeseries """
df_iod = pd.read_csv(iod_csv)
df_iod['date'] = pd.to_datetime(df_iod['datetime'])
df_iod = df_iod.drop("datetime",axis=1)
df_iod = df_iod.rename(columns={"date":"datetime"})
df_iod = df_iod.set_index("datetime")
## period
df_iod = df_iod.loc[startdate:enddate]

""" yield df"""
df_yield, df_yield_z = _yield_csv.main()
nondata_region = list(df_yield_z[df_yield_z.isna().all(axis=1)].index)
# max_yield = np.nanmax(df_yield.values)
# min_yield = np.nanmin(df_yield.values)

""" plot by region"""
def extract_df(df_yd_regi, df_val):
    mindate = df_yd_regi.index.min()
    maxdate = df_yd_regi.index.max()
    extracted_rows= df_val.loc[mindate:maxdate]
    return extracted_rows
    
    
for regi, row in tqdm(df_yield.iterrows()):
    yield_result = []
    for yer in row.index:
        dt_yield = datetime.datetime(int(yer),12,31)
        yield_val = row[yer]
        yield_result.append([dt_yield, yield_val])
        
    df_yield_regi = pd.DataFrame(yield_result, columns=["datetime","yield"])
    df_yield_regi = df_yield_regi.set_index("datetime")
    df_yield_regi = df_yield_regi.dropna()
       
    
    """ #Plot """
    fig,axes = plt.subplots(3,1, figsize=(10, 5))
    fig.subplots_adjust(hspace=0.5)
    ax = axes[0] #yield
    ax.scatter(df_yield_regi.index, df_yield_regi, color='black')
    ax.set_ylabel(f"Yield[ton/ha/yr]", fontsize = 14)
    
    ax = axes[1] #enso
    values = extract_df(df_yield_regi, df_enso_series)
    colors = ['red' if val > 0 else 'blue' for val in values.enso.values]
    ax.bar(values.index, values.enso.values,  color=colors, width=10)
    ax.axhline(0, color='gray', linewidth=0.7) #, linestyle='--'
    ax.set_ylabel(f"ENSO Index", fontsize = 14)
    
    ax = axes[2] #iod
    values = extract_df(df_yield_regi, df_iod)
    colors = ['red' if val > 0 else 'blue' for val in values.IOD.values]
    ax.bar(values.index, values.IOD.values,  color=colors, width=10)
    ax.axhline(0, color='gray', linewidth=0.7) #, linestyle='--'
    ax.set_ylabel(f"IOD Index", fontsize = 14)    
    
    # ax.tick_params(axis='x', labelsize=14)
    # ax.tick_params(axis='y', labelsize=14)
    # plt.xticks(high_var_r.index, x_label, rotation=45)
    # plt.legend(fontsize=14)
    plt.tight_layout()
    ### Export fig
    out_dir_fig = out_dir + os.sep + "_png"
    os.makedirs(out_dir_fig, exist_ok=True)
    regioname_fin = regi.replace(" ","")
    fig.savefig(out_dir_fig + os.sep + f"yield_enso_{regioname_fin}.png")
    plt.close()
    
                
            
            
        
        


