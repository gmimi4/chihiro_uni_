# -*- coding: utf-8 -*-
"""
Plot El Nino and El Nino period
"""
import os
import pandas as pd
import numpy as np
import calendar
import itertools
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt


enso_csv = r"F:\MAlaysia\ENSO\00_download\meiv2.csv"
out_dir = r"D:\Malaysia\02_Timeseries\YieldWater\05_var_variation_ENSO\_composite_plot_years_regimean\_png"
os.makedirs(out_dir, exist_ok=True)

startdate = datetime.datetime(2002,1,1)
enddate = datetime.datetime(2023,12,31)

mei_month_dict = {"DJ":1,"JF":2,"FM":3,"MA":4,"AM":5,"MJ":6,
                  "JJ":7,"JA":8,"AS":9,"SO":10,"ON":11,"ND":12}



def threshold_to_date(seri,thre):## threshold
    df_valid = seri.where(seri>thre) #|(seri<thre*-1)
    df_valid = df_valid.dropna()
    valid_date = list(df_valid.index)
    # convert date to end of month
    valid_date = [ts + pd.offsets.MonthEnd(0) for ts in valid_date]
    valid_date = list(set(valid_date))
    return valid_date

# --------------------------------------------
""" # all mei timeseries """
# --------------------------------------------
enso_thre = 0.5
df_mei = pd.read_csv(enso_csv)
df_mei = df_mei.set_index("YEAR")

df_mei_num = df_mei.rename(columns=mei_month_dict)

mei_result =[]
for yr,row in df_mei_num.iterrows():
    for m in row.index:
        datetime_ = datetime(int(yr), m, 1)
        mei_val= row.at[m]
        mei_result.append([datetime_, mei_val])
        
df_mei_series =pd.DataFrame(mei_result, columns=["datetime","enso"])
df_mei_series = df_mei_series.set_index("datetime")
## period
df_mei_series = df_mei_series.loc[startdate:enddate]


# --------------------------------------------
""" #  collect El Nino period (for 12 months)"""
# --------------------------------------------
""" # Identify months of ElNino, LaNiNa, Neutral"""

elnino_list, all_list = [],[]
for year, row in df_mei.iterrows():
    for colname, value in row.items():
        all_list.append([year, colname])
        if value >enso_thre:
            elnino_list.append([year, colname])
        

## filtering period
elnino_list = [e for e in elnino_list if (int(e[0])>=startdate.year)&(int(e[0])<=enddate.year)]
all_list = [e for e in all_list if (int(e[0])>=startdate.year)&(int(e[0])<=enddate.year)]

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


### ElNino from to #start from 0.5< till 1 year later
elnino_years = list(set([y.year for y in elnino_list_date]))
elnino_from_to =  []
for yr in elnino_years:
    elenino_months = [e for e in elnino_list_date if e.year==yr ]
    elenino_earliest = min(elenino_months)
    elenino_end = elenino_earliest + relativedelta(months=11)
    elnino_from_to.append([elenino_earliest, elenino_end])
    
#make throughout period
elnino_datetimes_ = [pd.date_range(start=period[0], end=period[1]).to_list() for period in elnino_from_to]
elnino_datetimes = []
for sublist in elnino_datetimes_:
    for dt in sublist:
        elnino_datetimes.append(dt)

elnino_datetimes_monthly = [d + pd.offsets.MonthEnd(0) for d in elnino_datetimes]
elnino_datetimes_monthly = list(set(elnino_datetimes_monthly))
elnino_datetimes_monthly = sorted(elnino_datetimes_monthly)
elnino_datetimes_monthly = pd.DatetimeIndex(elnino_datetimes_monthly)
elnino_datetimes_monthly = elnino_datetimes_monthly[
    (elnino_datetimes_monthly >= startdate) & (elnino_datetimes_monthly <= enddate)
]

# --------------------------------------------
""" #  Plot"""
# --------------------------------------------
""" # preparation for shade"""
month_diffs = elnino_datetimes_monthly.to_series().diff().dt.days // 30 #Compute differences in months
group_ids = elnino_datetimes_monthly[month_diffs.values>1] #find breakpoint datetime
elnino_datetimes_plot = []
for i,ti in enumerate(group_ids):
    if i ==0:
        start = elnino_datetimes_monthly[0]
        end_plus1 = group_ids[0]
        end_idx = elnino_datetimes_monthly.get_loc(end_plus1)
        end = elnino_datetimes_monthly[end_idx - 1]
    elif i == len(group_ids)-1:
        start = group_ids[i]
        end = elnino_datetimes_monthly[-1]
    else:
        start = ti
        end_plus1 = group_ids[i+1]
        end_idx = elnino_datetimes_monthly.get_loc(end_plus1)
        end = elnino_datetimes_monthly[end_idx - 1]
    elnino_datetimes_plot.append([start, end])
        
    
    


fig,ax = plt.subplots(figsize=(10, 5))
fig.subplots_adjust(hspace=0.5)
values = df_mei_series
colors = ['red' if val > 0 else 'blue' for val in df_mei_series.enso.values]
ax.bar(values.index, values.enso.values,  color=colors, width=10)
ax.axhline(0, color='gray', linewidth=0.7) #, linestyle='--'
first_legend = True
for interval in elnino_datetimes_plot:
    ax.axvspan(interval[0], interval[1], color='grey', alpha=0.3,
               label="El Nino period in this study" if first_legend else "")
    first_legend = False
ax.set_ylabel(f"ENSO Index", fontsize = 14)
ax.legend(loc="upper right")
plt.tight_layout()
### Export fig
out_dir_fig = out_dir + os.sep + "_png"
os.makedirs(out_dir_fig, exist_ok=True)
fig.savefig(out_dir_fig + os.sep + f"elnino_perido.png")
plt.close() 
        


