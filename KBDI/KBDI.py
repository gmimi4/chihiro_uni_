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
import copy
import math
import datetime
# from natsort import natsorted

with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
    rio_crs = CRS.from_epsg(4326)
    proj_crs = CRS.from_user_input(rio_crs)
    

#dailyからのmonthlyにしようかな
# 0.1 degree grid
# rain_annual_ave = r"D:\Malaysia\GPM\01_tif\_annual_sum_ave\IMERG_annual_sum_ave.tif"
# rain_tif_dir = r"D:\Malaysia\GPM\01_tif"
# temp_tif_dir = r"F:\MAlaysia\ECMWF\Temperature_2m\02_tif\daily"
# out_dir = r"D:\Malaysia\KBDI\1_daily"
rain_annual_ave = "/Volumes/SSD_2/Malaysia/GPM/01_tif_Affine/_annual_sum_ave/IMERG_annual_sum_ave.tif"
rain_tif_dir = "/Volumes/SSD_2/Malaysia/GPM/01_tif_Affine"
temp_tif_dir = "/Volumes/PortableSSD/MAlaysia/ECMWF/Temperature_2m/02_tif/daily_1950_resample_EVI"
out_dir = "/Volumes/SSD_2/Malaysia/KBDI/1_daily"

rain_tifs = glob.glob(rain_tif_dir+os.sep + "*tif")
temp_tifs = glob.glob(temp_tif_dir+os.sep + "*tif")

#sort関数は使用データによって調整して
# rain_tifs_sort = natsorted(rain_tifs, key=lambda y: int(os.path.basename(y)[:-4].split("_")[1]))
# temp_tifs_sort = natsorted(temp_tifs, key=lambda y: int(os.path.basename(y)[:-4]))
rain_tifs_sort = rain_tifs
temp_tifs_sort = temp_tifs

datelist = [os.path.basename(t)[:-4].split("_")[1] for t in temp_tifs_sort]
datelist = [datetime.datetime.strptime(t,'%Y%m%d') for t in datelist]
datelist.sort()

"""
# allign rasters #rain tifを小さい方（現状ではTemp）に合わせる
## arrayの位置は合っていると思うので出力はしないでArrayを処理
"""
temp_tif = temp_tifs_sort[0]

### annual rain tifをクロップ
src_rain_annual = rasterio.open(rain_annual_ave)
arr_rain_annual = src_rain_annual.read(1)
src_temp = rasterio.open(temp_tif)
arr_temp = src_temp.read(1)

#共通のmetaはここでアップデートする
meta = src_temp.meta
meta.update({"dtype" : "float64"})

xmin, ymin, xmax, ymax = src_temp.bounds
polygon_geom = Polygon([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
extent_polygon = gpd.GeoDataFrame(geometry=[polygon_geom], crs=proj_crs)

 ## mask larger raster by above polygon
annual_rain_clip, raster_transform = mask(src_rain_annual, extent_polygon.geometry, crop=True)

#右端columnがNoDataになっているので削除してshapeをTempに合わせる
if len(annual_rain_clip.shape)==3:
    annual_rain_clip = annual_rain_clip[0]

assert annual_rain_clip.shape == arr_temp.shape
# if annual_rain_clip.shape != arr_temp.shape:
#     annual_rain_clip = np.delete(annual_rain_clip[0], [-1], 1)
# else:
#     annual_rain_clip = annual_rain_clip[0]

src_rain_annual.close()
src_temp.close()
    
annual_rain_clip_inch = annual_rain_clip/25.4

#確認
# raster_transform_use = src_temp.transform
# meta=src_rain.meta.copy()
# meta.update({"transform":raster_transform_use,"width":raster_clip.shape[2],"height":raster_clip.shape[1],
#               "crs":proj_crs})
# path_to_output = out_dir +f"\\crop_check.tif"
# with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
#     with rasterio.open(path_to_output,"w",**meta) as dst:
#         dst.write(raster_clip)
    
"""
# 各日のtempとrainで処理開始
"""
# for i in tqdm(range(len(temp_tifs_sort))):
for i,date_time in tqdm(enumerate(datelist)):
    datee = datetime.datetime.strftime(date_time, '%Y%m%d')
    temp_tif = [t for t in temp_tifs_sort if datee in os.path.basename(t)[:-4]][0]
    filename_temp = os.path.basename(temp_tif)[:-4]
    # datee = filename_temp.split("_")[1]
    try:        
        rain_tif = [t for t in rain_tifs_sort if datee in os.path.basename(t)[:-4]][0]
    except:
        continue
    
    ### daily rain tifをクロップ
    with rasterio.open(rain_tif) as src_rain:
        arr_rain = src_rain.read(1)
        rain_clip, raster_transform = mask(src_rain, extent_polygon.geometry, crop=True)
    
    if len(rain_clip.shape)==3:
        rain_clip = rain_clip[0]
    assert rain_clip.shape == arr_temp.shape
    # if rain_clip.shape != arr_temp.shape:
    #     rain_clip = np.delete(rain_clip[0], [-1], 1)
    # else:
    #     rain_clip = rain_clip[0]
    
    """
    # Net rainレイヤー
    """
    #0.2以上のピクセルでは0.2をひく。0<0.2ではその値を継続、0は0
    #inchに換算
    rain_clip_inch = rain_clip/25.4
    
    #前日レイヤーがあること
    # if i ==0:
    if datee == datetime.datetime.strftime(datelist[0], '%Y%m%d'):
        net_rain_for_tomorrow = np.zeros(rain_clip_inch.shape)
    else:
        pass
    
    print(i)
    rain_clip_inch_new = rain_clip_inch + net_rain_for_tomorrow

    #該当するインデックスを取得する
    net_rain_indx_02 = np.where(rain_clip_inch_new >0.2) #0.2以上あった
    net_rain_indx_some = np.where((rain_clip_inch_new <0.2) & (rain_clip_inch_new>0)) #0.2未満 #andだとエラーがでる
    
    #indexを使って演算する
    net_rain_layer = copy.deepcopy(rain_clip_inch_new) #コピーしとく
    net_rain_layer[net_rain_indx_02] = rain_clip_inch[net_rain_indx_02]-0.2
    net_rain_layer[net_rain_indx_some] = 0 #for today's net rain but recover later for tomomorrow
    # use net_rain_layer for KBDI
    
    ##次の日以降用に0<0.2のピクセルは値復活させる
    net_rain_for_tomorrow = copy.deepcopy(rain_clip_inch_new)
    net_rain_for_tomorrow[net_rain_indx_02] = 0 #リセット
    net_rain_for_tomorrow[net_rain_indx_some] = rain_clip_inch_new[net_rain_indx_some]
    
    """
    # Tmaxレイヤー
    """
    #TmaxはTaveから推定する（Srinivasan, et al., 2001. Estimation of KBDI (Drought Index) in Real-Time Using GIS and Remote Sensing Technologies）
    # GHCNのTaveとTmaxとの直線式から
    Tmax_scale = 1.1
    arr_tmax = (arr_temp*Tmax_scale)*1.8 +32 #F
    
    """
    Q yesterday
    """
    # if i==0:
    if datee == datetime.datetime.strftime(datelist[0], '%Y%m%d'):
        # Q_yesterday = np.zeros(arr_temp.shape, dtype=float)  #Initialはゼロ
        Q_yesterday = np.full(arr_temp.shape, 400).astype(float)  #Initialは400
    else:
        pass
    
    """
    delta Q incliment
    """
    T = arr_tmax
    R = annual_rain_clip_inch
    Q = Q_yesterday
    #整数を足す箇所は整数のarrayをつくる
    array_1 = np.full(arr_tmax.shape, 1)
    array_083 = np.full(arr_tmax.shape, 0.83)
    array_800 = np.full(arr_tmax.shape, 800)
    
    deltaQ = ((array_800-Q)*(0.968*np.exp(0.0486*T)- array_083)*1)*0.001/(array_1+10.88*np.exp(-0.0441*R))
    
    """
    Q reduced
    """
    ## net rainがゼロ以上のところは*100して前日のQから引く
    net_rain100 = net_rain_layer*100
    Q_reduced =  Q_yesterday - net_rain100
    
    """
    Q today
    """
    Q_today = Q_reduced + deltaQ
    
    ### Export
    
    outfile = out_dir + os.sep + f"KBDI_{datee}.tif"
    with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
        with rasterio.open(outfile, "w",**meta) as dst:
            dst.write(Q_today,1)
    
    Q_yesterday = Q_today

