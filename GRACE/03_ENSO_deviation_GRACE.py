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
import datetime
from dateutil.relativedelta import relativedelta
import calendar
import itertools
from tqdm import tqdm
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as colors
from matplotlib import cm


tif_dir = r"F:\MAlaysia\GRACE\02_tif"
enso_csv = r"F:\MAlaysia\ENSO\00_download\meiv2.csv"
out_dir = r"F:\MAlaysia\GRACE\03_enso_devi"

# --------------------------------------
""" # create mean tif for all period
    # You do it once """
# --------------------------------------
mean_or_sum = "mean"
out_mean_dir = tif_dir + os.sep + f"yearly_{mean_or_sum}"
os.makedirs(out_mean_dir,exist_ok=True)
outfile_hint = f"GRACE_{mean_or_sum}"

tifs = glob.glob(tif_dir + os.sep + "*.tif")
tifs = [t for t in tifs if "thickness" in t]

arr_list = []
for tif in tifs:
    # tif = [t for t in tifs_ym if "2008363" in t][0]
    with rasterio.open(tif) as src:
        arr = src.read(1)
        meta = src.meta
        # arr = np.where(arr<-100,np.nan,arr) #Et
        arr_list.append(arr)

arr_stack =  np.array(arr_list)
if mean_or_sum =="mean":
    stack_stat = np.nanmean(arr_stack, axis=0)
elif mean_or_sum =="sum":
    stack_stat = np.nansum(arr_stack, axis=0)
else:
    print("nothing")

outfile = os.path.join(out_mean_dir, f"{outfile_hint}.tif")
# with rasterio.open(outfile, 'w', **meta) as dst:
#     dst.write(stack_stat,1)


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
""" # create mean tif for non enso period """
# --------------------------------------
all_date = []
for tif in tifs:
    tifdatestr = os.path.basename(tif)[:-4].split("_")[-2]
    tifdate = datetime.datetime.strptime(tifdatestr, '%Y%j')
    lastday = calendar.monthrange(tifdate.year, tifdate.month)[1]
    tifdate_last = datetime.datetime(tifdate.year, tifdate.month, lastday)
    all_date.append(tifdate_last)
    
def non_enso_tif(ellist):
    # el_date = elnino_date_monthly
    # el ="elnino"
    non_el_date = list(set(all_date) - set(ellist))
    
    """ # create non enso mean tif"""
    tifs_nonenso = []
    for non in non_el_date:
        for tif in tifs:
            tifdatestr = os.path.basename(tif)[:-4].split("_")[-2]
            tifdate = datetime.datetime.strptime(tifdatestr, '%Y%j')
            if (tifdate.year==non.year and tifdate.month==non.month):
                tifs_nonenso.append(tif)
        
    arrs_non = []
    for t in tifs_nonenso:
        with rasterio.open(t) as src:
            arr = src.read(1)
            arrs_non.append(arr)
    
    arr_stack = np.stack(arrs_non)
    arr_non_mean = np.nanmean(arr_stack, axis=0) #seasonal mean
    
    return arr_non_mean


arr_nonel_mean = non_enso_tif(elnino_date_monthly)
arr_nonla_mean = non_enso_tif(lanina_date_monthly)

out_nonel_tif = out_mean_dir + os.sep + "non_elnino_mean.tif"
with rasterio.open(out_nonel_tif, "w", **meta) as dst:
    dst.write(arr_nonel_mean, 1)
    
out_nonel_tif = out_mean_dir + os.sep + "non_lanina_mean.tif"
with rasterio.open(out_nonel_tif, "w", **meta) as dst:
    dst.write(arr_nonla_mean, 1)



# --------------------------------------
""" # Seasonal deviation from global mean """
# --------------------------------------
seasons = {"DJF":[12,1,2],"MAM":[3,4,5],"JJA":[6,7,8],"SON":[9,10,11]}

with rasterio.open(outfile) as src:
    arr_global_mean = src.read(1)
    meta = src.meta

def generate_devitif(el_date, el):
    # el_date = elnino_date_monthly
    # el ="elnino"
    deviations = []
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
                arr = src.read(1)
                arrs_seas.append(arr)
        
        arr_stack = np.stack(arrs_seas)
        arr_seas_mean = np.nanmean(arr_stack, axis=0) #seasonal mean
        
        """ # deviation by seasons"""
        arr_devi = arr_seas_mean - arr_global_mean
        deviations.append(arr_devi)
        
    
    arr_stack_devi = np.stack(deviations)
    arr_min = np.nanmin(arr_stack_devi, axis=0)
    arr_max = np.nanmax(arr_stack_devi, axis=0)
    arr_devi_mean = np.nanmean(arr_stack_devi, axis=0)
    
    outtif_names = {f"{el}_min":arr_min,f"{el}_max":arr_max,f"{el}_mean":arr_devi_mean}
    for outname, ar in outtif_names.items():
        outfile_devi = out_dir + os.sep + f"devi_{outname}.tif"
        with rasterio.open(outfile_devi, 'w', **meta) as dst:
            dst.write(ar,1)
            
            
### Process
generate_devitif(elnino_date_monthly, "elnino")
generate_devitif(lanina_date_monthly, "lanina")



# --------------------------------------
""" # deviation from non enso mean """
# --------------------------------------

def generate_devitif_nonenso(el_date, el, non_arr):
    # el_date = elnino_date_monthly
    # el ="elnino"
    
    """ # deviation by seasons"""
    deviations = []
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
            
        ### create seasonal mean tif"""
        arrs_seas = []
        for t in tifs_seas:
            with rasterio.open(t) as src:
                arr = src.read(1)
                arrs_seas.append(arr)
        
        arr_stack = np.stack(arrs_seas)
        arr_seas_mean = np.nanmean(arr_stack, axis=0) #seasonal mean
        
        arr_devi = arr_seas_mean - non_arr
        deviations.append(arr_devi)
       
        outtif_names = f"devi_{el}_{seas}_bynon.tif"
        outfile_devi = out_dir + os.sep + outtif_names
        with rasterio.open(outfile_devi, 'w', **meta) as dst:
            dst.write(arr_devi,1)
    
    
    """ # deviation by non enso all period"""
    el_year_month = [(ts.year, ts.month) for ts in el_date]
    
    enso_tifs = []
    for ym in el_year_month:  
        for tif in tifs:
            tifdatestr = os.path.basename(tif)[:-4].split("_")[-2]
            tifdate = datetime.datetime.strptime(tifdatestr, '%Y%j')
            if (tifdate.year==ym[0] and tifdate.month==ym[1]):
                enso_tifs.append(tif)
            
    ### create enso mean tif
    arrs_enso = []
    for t in enso_tifs:
        with rasterio.open(t) as src:
            arr = src.read(1)
            arrs_enso.append(arr)
    
    arr_stack = np.stack(arrs_enso)
    arr_enso_mean = np.nanmean(arr_stack, axis=0) #enso mean
    
    ### deviation by seasons"""
    arr_devi = arr_enso_mean - non_arr
    
    outtif_names = f"{el}_mean_bynon"
    outfile_devi = out_dir + os.sep + f"devi_{outtif_names}.tif"
    with rasterio.open(outfile_devi, 'w', **meta) as dst:
        dst.write(arr_devi,1)
            
            
### Process
generate_devitif_nonenso(elnino_date_monthly, "elnino", arr_nonel_mean)
generate_devitif_nonenso(lanina_date_monthly, "lanina", arr_nonla_mean)



# ---------------
""" # Plot"""
# ---------------
import geopandas as gpd
from rasterio.windows import from_bounds
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter

# shp_region = '/Volumes/SSD_2/Malaysia/Validation/1_Yield_doc/shp/region_slope_fin.shp'
shp_region = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
# tif = '/Volumes/PortableSSD/Malaysia/ENSO/01_deviations/EVI/elnino_devi_all.tif'
gdf = gpd.read_file(shp_region)
# gdf = gdf.to_crs(raster_crs)

min_lon = 93
min_lat = -12
max_lon = 142
max_lat = 9

def format_lon(x, pos):
        return f"{abs(int(x))}°{'E' if x >= 0 else 'W'}"

def format_lat(y, pos):
    return f"{abs(int(y))}°{'N' if y >= 0 else 'S'}"

def plot_devi(tif_tar, tif_sub, mean_devi):
    out_png_dir = os.path.dirname(tif_tar) + os.sep + "_png"
    os.makedirs(out_png_dir, exist_ok=True)
    """ # obtain data range """
    with rasterio.open(tif_tar) as src:
        transform = src.transform
        raster_crs = src.crs
        window = from_bounds(min_lon, min_lat, max_lon, max_lat, transform=src.transform)
        arr_tar = src.read(1, window=window)
        bounds = rasterio.windows.bounds(window, src.transform)
        arr_tar_1d = np.ravel(arr_tar)
        arr_tar_1d = arr_tar_1d[~np.isnan(arr_tar_1d)]
        # minval = np.nanmin(arr)
        # maxval = np.nanmax(arr)
        minval_1 = np.percentile(arr_tar_1d, 5)
        maxval_1 = np.percentile(arr_tar_1d, 95)
   
    with rasterio.open(tif_sub) as src:
       transform = src.transform
       raster_crs = src.crs
       window = from_bounds(min_lon, min_lat, max_lon, max_lat, transform=src.transform)
       arr_sub = src.read(1, window=window)
       bounds = rasterio.windows.bounds(window, src.transform)
       arr_1d_sub = np.ravel(arr_sub)
       arr_1d_sub = arr_1d_sub[~np.isnan(arr_1d_sub)]
       # minval = np.nanmin(arr)
       # maxval = np.nanmax(arr)
       minval_2 = np.percentile(arr_1d_sub, 5)
       maxval_2 = np.percentile(arr_1d_sub, 95)
   
    ### comparison 
    minval = min([minval_1, minval_2])
    maxval = max([maxval_1, maxval_2])
    norm = TwoSlopeNorm(vmin=minval, vcenter=0, vmax=maxval)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = cm.get_cmap('coolwarm_r').copy()
    cmap.set_bad(color='none')  # na
    img = ax.imshow(arr_tar, cmap='coolwarm_r', norm=norm, extent=[bounds[0], bounds[2], bounds[1], bounds[3]], origin='upper')
    gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(img, cax=cax, shrink=0.8, orientation='vertical', pad=0.07) #ax=ax,
    if mean_devi =="devi":
        cbar.set_label("GRACE anomaly \n[cm]", fontsize=25)
    if mean_devi =="mean":
        cbar.set_label("GRACE mean \n[cm]", fontsize=25)
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_png_dir + os.sep + f'{os.path.basename(tif_tar)[:-4]}.png', dpi=600)
    plt.close()
    

meandevi = "devi"
tifel = r"F:\MAlaysia\GRACE\03_enso_devi\devi_elnino_mean_bynon.tif"
tifla = r"F:\MAlaysia\GRACE\03_enso_devi\devi_lanina_mean_bynon.tif"
plot_devi(tifel, tifla, meandevi)      
plot_devi(tifla, tifel, meandevi)

tifel = r"F:\MAlaysia\GRACE\03_enso_devi\devi_elnino_DJF_bynon.tif"
tifla = r"F:\MAlaysia\GRACE\03_enso_devi\devi_lanina_DJF_bynon.tif"
plot_devi(tifel, tifla, meandevi)      
plot_devi(tifla, tifel, meandevi)

tifel = r"F:\MAlaysia\GRACE\03_enso_devi\devi_elnino_JJA_bynon.tif"
tifla = r"F:\MAlaysia\GRACE\03_enso_devi\devi_lanina_JJA_bynon.tif"
plot_devi(tifel, tifla, meandevi)      
plot_devi(tifla, tifel, meandevi)
            
tifel = r"F:\MAlaysia\GRACE\03_enso_devi\devi_elnino_MAM_bynon.tif"
tifla = r"F:\MAlaysia\GRACE\03_enso_devi\devi_lanina_MAM_bynon.tif"
plot_devi(tifel, tifla, meandevi)      
plot_devi(tifla, tifel, meandevi)  

tifel = r"F:\MAlaysia\GRACE\03_enso_devi\devi_elnino_SON_bynon.tif"
tifla = r"F:\MAlaysia\GRACE\03_enso_devi\devi_lanina_SON_bynon.tif"
plot_devi(tifel, tifla, meandevi)      
plot_devi(tifla, tifel, meandevi)    

### mean
meandevi = "mean"
tifel = r"F:\MAlaysia\GRACE\03_enso_mean\elnino_mean.tif"
tifla = r"F:\MAlaysia\GRACE\03_enso_mean\lanina_mean.tif"
plot_devi(tifel, tifla, meandevi)      
plot_devi(tifla, tifel, meandevi)


tifel = r"F:\MAlaysia\GRACE\03_enso_mean\elnino_mean_DJF.tif"
tifla = r"F:\MAlaysia\GRACE\03_enso_mean\lanina_mean_DJF.tif"
plot_devi(tifel, tifla, meandevi)
plot_devi(tifla, tifel, meandevi)
  
tifel = r"F:\MAlaysia\GRACE\03_enso_mean\elnino_mean_MAM.tif"
tifla = r"F:\MAlaysia\GRACE\03_enso_mean\lanina_mean_MAM.tif"
plot_devi(tifel, tifla, meandevi)
plot_devi(tifla, tifel, meandevi)

tifel = r"F:\MAlaysia\GRACE\03_enso_mean\elnino_mean_SON.tif"
tifla = r"F:\MAlaysia\GRACE\03_enso_mean\lanina_mean_SON.tif"
plot_devi(tifel, tifla, meandevi)
plot_devi(tifla, tifel, meandevi)

tifel = r"F:\MAlaysia\GRACE\03_enso_mean\elnino_mean_JJA.tif"
tifla = r"F:\MAlaysia\GRACE\03_enso_mean\lanina_mean_JJA.tif"
plot_devi(tifel, tifla, meandevi)
plot_devi(tifla, tifel, meandevi)

### simple GRACE val
tifel = r"F:\MAlaysia\GRACE\02_tif\yearly_mean\GRACE_mean_allperiod.tif"
tifla = r"F:\MAlaysia\GRACE\02_tif\yearly_mean\GRACE_mean_allperiod.tif"
plot_devi(tifel, tifla, meandevi)


