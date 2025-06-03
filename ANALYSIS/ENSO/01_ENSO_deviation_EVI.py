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
os.chdir(r"C:\Users\chihiro\Desktop\Python\ANALYSIS\ENSO")
import _ENSOperiod


tif_dir = r"F:\MAlaysia\GRACE\02_tif"
enso_csv = r"F:\MAlaysia\ENSO\00_download\meiv2.csv"
out_dir = r"F:\MAlaysia\GRACE\03_enso_mean"

tifs = glob.glob(tif_dir + os.sep + "*.tif")
tifs = [t for t in tifs if "thickness" in t]

# --------------------------------------
""" # Find ENSO period """
# --------------------------------------
elnino_date_monthly, lanina_date_monthly = _ENSOperiod.ensolist()


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

