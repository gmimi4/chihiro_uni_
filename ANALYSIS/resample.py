# -*- coding: utf-8 -*-
"""Ndaysmean_and_Resample.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wfwrytmnAQfX7CrRHX8HLu9fHPPCZ-GR

n days meanの計算とResample
"""

import os,sys
import glob
import rasterio
# import xarray
# import rioxarray
import numpy as np
import tqdm
from rasterio.merge import merge
from rasterio.plot import show
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio.crs import CRS
from pyproj.crs import CRS
import math
from tqdm import tqdm

with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
    rio_crs = CRS.from_epsg(4326)
    proj_crs = CRS.from_user_input(rio_crs)


# in_dir_path = sys.argv[1]
in_dir_path = r"D:\Malaysia\Validation\2_PALSAR_mosaic\00_download\GEE"
out_dir = r"D:\Malaysia\Validation\2_PALSAR_mosaic\01_res100m"
# in_dir_path = r"F:\MAlaysia\GLEAM\02_tif\Et"
# out_dir = r"F:\MAlaysia\GLEAM\02_tif\Et\res_01"
os.makedirs(out_dir,exist_ok=True)
# in_dir_path = r'D:\Malaysia\ECMWF\Temperature_2m\monthly\02_tif'
# out_dir = r"F:\MAlaysia\ECMWF\Temperature_2m\05_monthly\025grid"
# in_dir_path = r'D:\Malaysia\GSMap\02_tif\monthly'
# out_dir = r"F:\MAlaysia\GSMap\04_sum\monthly"

tif_list = glob.glob(in_dir_path+"\\*.tif")
# tif_list = [t for t in tif_list if "GOSIF_2023" in t]
outname_extent = "_res100.tif"
outname_extent = "500m.tif"

#------------------------------------------------
"""Resample"""
#------------------------------------------------

new_resolution = (0.001, 0.001)
new_resolution = (0.05, 0.05)
#以下のコードはGPPでは良いと思うが、Temperatureでは0を含めて?よくない　その下のColabでうまくいったコードを採用している。
"""
#Gpp用にする
for input_raster_path in tqdm(tif_list):
    # input_raster_path = r"F:\MAlaysia\GSMap\02_tif\8daysmean_grid0.1\20100101_sum_gsm.tif"
    
    # src=rasterio.open(input_raster_path)
    ## Bilinerがおかしくなるのでnanを0に置換する ### ###追記：これするとゼロを含めてしまう 未解決###
    with rasterio.open(input_raster_path) as src: #nodata=None
        data = src.read(1)
        data[np.isnan(data)] = 0
        # nodata=np.nan
        profile = src.profile
        profile.update(crs=proj_crs)
        
        tmp_file = out_dir + "/"+ os.path.basename(input_raster_path)[:-4]+"_0.tif"
        with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
            with rasterio.open(tmp_file, 'w', **profile) as dst:
              dst.write(data,1)
      
    with rasterio.open(tmp_file) as src:   
       data = src.read(
           out_shape=(src.count, math.ceil(src.height*src.res[0]/new_resolution[0]),math.floor(src.width * src.res[1] / new_resolution[1])),
                      resampling = Resampling.average) #nearest #bilinear #.average
       profile = src.profile
       profile.update(width=data.shape[2],height =data.shape[1], crs=proj_crs, transform = rasterio.transform.from_origin(
           src.bounds.left, src.bounds.top, new_resolution[0], new_resolution[1]
       ))
      
       out_file = out_dir + "/"+ os.path.basename(input_raster_path)[:-4]+ outname_extent
       with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
          with rasterio.open(out_file, 'w', **profile) as dst:
            dst.write(data[0],1)
        
    os.remove(tmp_file)
"""    
for input_raster_path in tqdm(tif_list):
    # input_raster_path = tif_list[0]
    src = rasterio.open(input_raster_path)
    arr = src.read(1)
    
    data = src.read(
        out_shape=(src.count, int(src.height*src.res[0]/new_resolution[0]),int(src.width * src.res[1] / new_resolution[1])),
                  resampling = Resampling.nearest) #nearest: when to finer, #bilinear: when to coraser
    profile = src.profile
    profile.update(crs=proj_crs, width=data.shape[2],height =data.shape[1], transform = rasterio.transform.from_origin(
        src.bounds.left, src.bounds.top, new_resolution[0], new_resolution[1]
    ))
    src.close()

    # Export raster
    out_file = out_dir + "/"+ os.path.basename(input_raster_path)[:-4]+ outname_extent
    with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
        with rasterio.open(out_file, 'w', **profile) as dst:
          dst.write(arr, 1)
          
# #Colabでうまくいったコード↓
# #参照用
# with rasterio.open(tif_list[0]) as src:
#   data = src.read(
#       out_shape=(src.count, int(src.height*src.res[0]/new_resolution[0]),int(src.width * src.res[1] / new_resolution[1])),
#                 resampling = Resampling.nearest) #nearest: when to finer, #bilinear: when to coraser
#   profile = src.profile
#   profile.update(crs=proj_crs, width=data.shape[2],height =data.shape[1], transform = rasterio.transform.from_origin(
#       src.bounds.left, src.bounds.top, new_resolution[0], new_resolution[1]
#   ))

# #処理
# for input_raster_path in tqdm(tif_list):
#     # input_raster_path = tif_list[0]
#     src = rasterio.open(input_raster_path)
#     arr = src.read()[0]
#     src.close()

#     # Export raster
#     out_file = out_dir + "/"+ os.path.basename(input_raster_path)[:-4]+ outname_extent
#     with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
#         with rasterio.open(out_file, 'w', **profile) as dst:
#           dst.write(arr, 1)

