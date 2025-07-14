# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:57:00 2024

@author: chihiro

KBDI index by: A Drought Index for Forest Fire Control (Keetch, 1968)
"""

import os, sys
import rasterio
import numpy as np
import glob
from shapely.geometry import Polygon
import geopandas as gpd
from rasterio.mask import mask
from rasterio.windows import Window
from rasterio.crs import CRS
from pyproj.crs import CRS
from tqdm import tqdm

with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
    rio_crs = CRS.from_epsg(4326)
    proj_crs = CRS.from_user_input(rio_crs)
    

#dailyからのmonthlyにしようかな
# 0.1 degree grid
# rain_tif_dir = r"D:\Malaysia\GPM\01_tif"
# out_dir = r"D:\Malaysia\GPM\01_tif\_annual_sum_ave"
rain_tif_dir = r"/Volumes/SSD_2/Malaysia/GPM/01_tif_Affine"
out_dir = rain_tif_dir + os.sep + "_annual_sum_ave"
os.makedirs(out_dir, exist_ok=True)

rain_tifs = glob.glob(rain_tif_dir+os.sep + "*tif")


### Obtain anual mean (sum) rain
years_list = [y for y in range(2000,2024,1)]
rain_tifs_years = {}
for year in years_list:
    rain_tifs_year = []
    for tif in rain_tifs:
        date_str = os.path.basename(tif)[:-4].split("_")[1]
        year_str = date_str[0:4]
        if year_str == str(year):
            rain_tifs_year.append(tif)
    
    rain_tifs_years[str(year)] = rain_tifs_year
    
#各年の合計降水量
rain_sum_years ={}
for year, tif_list in tqdm(rain_tifs_years.items()):
    # tif_list=rain_tifs_years["2022"]
    arr_list = []
    for tif in tif_list:
        src = rasterio.open(tif)
        arr = src.read(1)
        arr_list.append(arr)
    
    # Compute anything what you want
    arr_stack = np.stack(arr_list)
    array_sum = np.nansum(arr_list, axis=0)
    rain_sum_years[year]=array_sum

#上記の平均値を得る
all_sum_arr = [arr for year,arr in rain_sum_years.items()]
sumarr_stack = np.stack(all_sum_arr)
array_ave = np.nanmean(all_sum_arr, axis=0)

#Export
out_path = out_dir + os.sep + "IMERG_annual_sum_ave.tif"
sample_src = rasterio.open(rain_tifs[0])
meta = sample_src.meta

with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(array_ave,1)
    






