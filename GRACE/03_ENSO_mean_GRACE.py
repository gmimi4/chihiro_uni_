# -*- coding: utf-8 -*-
"""
#1. simply mean in ENSO period
"""

import numpy as np
import pandas as pd
import os,sys
import glob
from datetime import datetime
import datetime
from dateutil.relativedelta import relativedelta
import calendar
import itertools
from tqdm import tqdm
import rasterio


tif_dir = r"F:\MAlaysia\GRACE\02_tif"
enso_csv = r"F:\MAlaysia\ENSO\00_download\meiv2.csv"
out_dir = r"F:\MAlaysia\GRACE\03_enso_mean"

tifs = glob.glob(tif_dir + os.sep + "*.tif")
tifs = [t for t in tifs if "thickness" in t]

# --------------------------------------
""" # Find ENSO period """
# --------------------------------------
month_calendar = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec" }
mei_month_dict = {"DJ":1,"JF":2,"FM":3,"MA":4,"AM":5,"MJ":6,
                  "JJ":7,"JA":8,"AS":9,"SO":10,"ON":11,"ND":12}

startdate = datetime.datetime(2002,1,1)
enddate = datetime.datetime(2023,12,31)

### Extract date when exceeding threhold
def threshold_to_date(seri,thre):## threshold
    df_valid = seri.where(seri>thre) #|(seri<thre*-1)
    df_valid = df_valid.dropna()
    valid_date = list(df_valid.index)
    # convert date to end of month
    valid_date = [ts + pd.offsets.MonthEnd(0) for ts in valid_date]
    valid_date = list(set(valid_date))
    return valid_date


""" # Identify months of ElNino, LaNiNa"""
enso_thre = 0.5
df_mei = pd.read_csv(enso_csv)
df_mei = df_mei.set_index("YEAR")

elnino_list_, lanina_list = [],[]

for year, row in df_mei.iterrows():
    for colname, value in row.items():
        if value >enso_thre:
            elnino_list_.append([year, colname])
        elif value < -1 * enso_thre:
            lanina_list.append([year, colname])
        else:
            pass

""" # convert mei str to datetime """
def convert_mei_date(mei_list):
    mei_list_date = []
    for me in mei_list:
        yearint = int(me[0])
        mei_mon = mei_month_dict[me[1]]
        lastday1 = calendar.monthrange(yearint, mei_mon) #num of weeks, days
        yyyymmdd_1 =  datetime.datetime(yearint, mei_mon, lastday1[1])
        mei_list_date.append([yyyymmdd_1])
    
    mei_list_date = list((itertools.chain.from_iterable(mei_list_date)))
    mei_list_date = sorted(list(set(mei_list_date)))
    
    return mei_list_date


def monthly_enso_date(elnino_list):
    ## filtering period
    elnino_list = [e for e in elnino_list if (int(e[0])>=startdate.year)&(int(e[0])<=enddate.year)]
    											    
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
    
    return elnino_datetimes_monthly
 

elnino_date_monthly = monthly_enso_date(elnino_list_) #datetime index
lanina_date_monthly = monthly_enso_date(lanina_list)

elnino_date_monthly = list(elnino_date_monthly.to_pydatetime()) #convert to datetime list
lanina_date_monthly = list(lanina_date_monthly.to_pydatetime()) #convert to datetime list

# --------------------------------------
""" # create mean tif for enso period all & by seasons """
# --------------------------------------
""" # Seasonal deviation from global mean """

seasons = {"DJF":[12,1,2],"MAM":[3,4,5],"JJA":[6,7,8],"SON":[9,10,11]}

def generate_devitif(el_date, el):
    # el_date = elnino_date_monthly
    # el ="elnino"
    arrs_allenso = []
    for seas,sealist in seasons.items():
        el_date_season = [e for e in el_date if e.month in sealist]
        el_year_month = [(ts.year, ts.month) for ts in el_date_season]
        tifs_seas = []
        for ym in el_year_month:
            for tif in tifs:
                tifdatestr = os.path.basename(tif)[:-4].split("_")[-2]
                tifdate = datetime.datetime.strptime(tifdatestr, '%Y%j')
                if (tifdate.year==ym[0] and tifdate.month==ym[1]):
                    tifs_seas.append(tif)
            
        """ # create seasonal mean tif"""
        arrs_seas = []
        for t in tifs_seas:
            with rasterio.open(t) as src:
                meta = src.meta
                arr = src.read(1)
                arrs_seas.append(arr)
        
        arr_stack = np.stack(arrs_seas)
        arr_seas_mean = np.nanmean(arr_stack, axis=0) #seasonal mean
        
        arrs_allenso.append(arr_seas_mean)
    
        outfile = out_dir + os.sep + f"{el}_mean_{seas}.tif"
        with rasterio.open(outfile, 'w', **meta) as dst:
            dst.write(arr_seas_mean,1)
            
    """ # create all period mean tif"""
    arr_stack_all = np.stack(arrs_allenso)
    arr_all_mean = np.nanmean(arr_stack_all, axis=0) #seasonal mean
    outfile = out_dir + os.sep + f"{el}_mean.tif"
    with rasterio.open(outfile, 'w', **meta) as dst:
        dst.write(arr_all_mean,1)
            
            
### Process
generate_devitif(elnino_date_monthly, "elnino")
generate_devitif(lanina_date_monthly, "lanina")

