# -*- coding: utf-8 -*-
"""
#1. simply mean in ENSO period
# ENSO period is defined from threhold for 11 months.
"""

import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as colors
from matplotlib import cm
# os.chdir(r"C:\Users\chihiro\Desktop\Python\ANALYSIS\ENSO")
os.chdir("/Users/wtakeuchi/Desktop/Python/ANALYSIS/ENSO")
import _ENSOperiod


tif_dir = "/Volumes/PortableSSD/Malaysia/MODIS_EVI/01_MOD13A2061_resample/_4326_res01_age_adjusted"
out_dir = "/Volumes/PortableSSD/Malaysia/ENSO/01_deviations/EVI"

tifs = glob.glob(tif_dir + os.sep + "*.tif")

# --------------------------------------
""" # Find ENSO period """
# --------------------------------------
elnino_date_monthly, lanina_date_monthly = _ENSOperiod.ensolist()

ENSO_date_dic = {'elnino':elnino_date_monthly, 'lanina':lanina_date_monthly}

# --------------------------------------
""" # create mean tif for enso period """
# --------------------------------------

seasons = {"DJF":[12,1,2],"MAM":[3,4,5],"JJA":[6,7,8],"SON":[9,10,11]}

def generate_seasonal_meanras(datelist, seastr):
    
    # for seas,sealist in seasons.items():
    el_date_season = [e for e in datelist if e.month in seasons[seastr]]
    el_year_month = [(ts.year, ts.month) for ts in el_date_season]
    tifs_seas = [] #ENSO tif
    for ym in el_year_month:
        for tif in tifs:
            tifdatestr = os.path.basename(tif)[:-4].split("_")[1]
            tifdate = datetime.datetime.strptime(tifdatestr, '%Y%m%d')
            if (tifdate.year==ym[0] and tifdate.month==ym[1]):
                tifs_seas.append(tif)
        
    """ # create seasonal array"""
    arrs_seas = []
    for t in tifs_seas:
        with rasterio.open(t) as src:
            meta = src.meta
            arr = src.read(1)
            arr_re = np.where(arr==0, np.nan, arr) 
            arrs_seas.append(arr_re)

    arr_stack = np.stack(arrs_seas)
    arr_seas_mean_ = np.nanmean(arr_stack, axis=0) #seasonal mean
    
    return arr_seas_mean_



for el, el_date in ENSO_date_dic.items():
    # el_date = elnino_date_monthly
    # el ="elnino"
    """ # create full daily datetime list"""
    date_list = pd.date_range(start="2002-01-01", end="2023-12-31", freq="ME").to_list()
    datetime_list = [dt.to_pydatetime() for dt in date_list]
    nonel_date = set(datetime_list)^set(el_date)
    
    arrs_allseason = []
    arrs_allseason_non = []
    for seas,sealist in seasons.items():
        arr_seas_mean = generate_seasonal_meanras(el_date, seas) *0.0001 #seasonal mean
        arr_seas_mean_non = generate_seasonal_meanras(nonel_date, seas) *0.0001 #seasonal mean nonENSO
        """ # seasonal deviation"""
        arr_seas_devi = (arr_seas_mean - arr_seas_mean_non)
        
        arrs_allseason.append(arr_seas_mean)
        arrs_allseason_non.append(arr_seas_mean_non)
    
        outfile = out_dir + os.sep + f"{el}_devi_{seas}.tif"
        with rasterio.open(tifs[0]) as src: #sample
            meta = src.meta
            meta.update(dtype='float32')
        with rasterio.open(outfile, 'w', **meta) as dst:
            dst.write(arr_seas_mean,1)
            
    """ # create all period mean tif"""
    arr_stack_all = np.stack(arrs_allseason)
    arr_stack_all_non = np.stack(arrs_allseason_non)
    arr_all_mean = np.nanmean(arr_stack_all, axis=0) #all mean
    arr_all_mean_non = np.nanmean(arr_stack_all_non, axis=0) #all mean
    arr_all_devi = arr_all_mean - arr_all_mean_non
    outfile = out_dir + os.sep + f"{el}_devi_all.tif"
    with rasterio.open(outfile, 'w', **meta) as dst:
        dst.write(arr_all_devi,1)
            
            


# ---------------
""" # Plot"""
# ---------------
shp_region = '/Volumes/SSD_2/Malaysia/Validation/1_Yield_doc/shp/region_slope_fin.shp'
# tif = '/Volumes/PortableSSD/Malaysia/ENSO/01_deviations/EVI/elnino_devi_all.tif'
gdf = gpd.read_file(shp_region)
# gdf = gdf.to_crs(raster_crs)


def plot_devi(tif):
    out_dir = os.path.dirname(tif) + os.sep + "_png"
    """ # obtain data range """
    with rasterio.open(tif) as src:
        bounds = src.bounds
        transform = src.transform
        raster_crs = src.crs
        arr = src.read(1)
        arr_1d = np.ravel(arr)
        arr_1d = arr_1d[~np.isnan(arr_1d)]
        # minval = np.nanmin(arr)
        # maxval = np.nanmax(arr)
        minval = np.percentile(arr_1d, 5)
        maxval = np.percentile(arr_1d, 95)
    norm = TwoSlopeNorm(vmin=minval, vcenter=0, vmax=maxval)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = cm.get_cmap('coolwarm').copy()
    cmap.set_bad(color='none')  # na
    img = ax.imshow(arr, cmap='coolwarm', norm=norm, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], origin='upper')
    gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
    cbar = plt.colorbar(img, ax=ax, shrink=0.7, orientation='horizontal', pad=0.07)
    cbar.set_label("EVI anomaly [-]", fontsize=25)
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_dir + os.sep + f'{os.path.basename(tif)[:-4]}.png', dpi=600)
    

plot_devi('/Volumes/PortableSSD/Malaysia/ENSO/01_deviations/EVI/elnino_devi_all.tif')      
plot_devi('/Volumes/PortableSSD/Malaysia/ENSO/01_deviations/EVI/lanina_devi_all.tif')        

