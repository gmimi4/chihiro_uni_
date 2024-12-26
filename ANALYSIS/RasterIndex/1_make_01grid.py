# -*- coding: utf-8 -*-
"""
Make 0.1 degree grid from sample tif
SIF／EVI以外のラスターでインデックスがずれていそうなため確認のため作成する
"""
# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install factor_analyzer

import numpy as np
import pandas as pd
import os,sys
import rasterio
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
# import pyogrio
from rasterstats import zonal_stats
# from osgeo import gdal, ogr
# import matplotlib.pyplot as plt
from tqdm import tqdm

# PageName = sys.argv[1]
PageName = "A4"
# sample_tif = f'/Volumes/SSD_2/Malaysia/02_Timeseries/CPA_CPR/2_out_ras/p_01/{PageName}_Eb_importance_2002-2012.tif'
out_dir = '/Volumes/SSD_2/Malaysia/02_Timeseries/99_rasterIndex'

def make01grid(sample_tif):
    src = rasterio.open(sample_tif)
    src_arr = src.read(1)
    h = src_arr.shape[0]
    w = src_arr.shape[1]
    src_arr_tmp = np.arange(h*w).reshape(h, w).astype("int16")
    
    """ #0.1 degree gridポリゴンをつくる 
         use already existed one"""
    
    rasname = os.path.basename(sample_tif)[:-4]
    outgrid_name =  f"grid_01degree_{rasname}.shp"
    # if not os.path.isfile(out_dir + os.sep +outgrid_name):
    # Generate polygons from the raster data
    mask = None
    results = (
    {'properties': {'raster_val': v}, 'geometry': s}
    for i, (s, v) 
    in enumerate(
        shapes(src_arr_tmp, mask=mask, transform=src.transform)))
    
    #Create geopandas Dataframe and save as geojson, ESRI shapefile etc.
    geoms = list(results)
    gpd_polygonized_raster  = gpd.GeoDataFrame.from_features(geoms)
    gpd_polygonized_raster["area"] = gpd_polygonized_raster.geometry.area
    gpd_polygonized_raster = gpd_polygonized_raster.set_crs(4326)
    grid_area = f'{gpd_polygonized_raster.iloc[1].area:.2f}' #'0.01 degree^2' = 100km2 = 100*10^6
    #check
    gpd_polygonized_raster.to_file(os.path.join(out_dir, outgrid_name), crs="epsg:4326")

        
    



sample_tif_list = [f'/Volumes/PortableSSD/Malaysia/ECMWF/Temperature_2m/02_tif/daily_1950/extent/ERA51950_20000110_extentafter_{PageName}.tif',
                   f'/Volumes/SSD_2/Malaysia/GPM/01_tif_Affine/extent/MERG_20210521_affine_extentafter_{PageName}.tif',
                   f'/Volumes/PortableSSD/Malaysia/ECMWF/VPD/extent/VPD_20190526_extentafter_{PageName}.tif',
                   f'/Volumes/PortableSSD/Malaysia/GLEAM/02_tif_v41/Eb/extent/Eb_2000001_extentafter_{PageName}.tif',
                   f'/Volumes/PortableSSD/Malaysia/GLEAM/02_tif_v41/Et/extent/Et_2000001_extentafter_{PageName}.tif',
                   f'/Volumes/PortableSSD 1/Malaysia/AMSRE/01_tif/SM_C/sameday/extent/AMSRE_D_SM_C_20020619_extentafter_{PageName}.tif',
                   f'/Volumes/PortableSSD 1/Malaysia/AMSRE/01_tif/VOD_C/sameday/extent/AMSRE_D_VOD_C_20020619_extentafter_{PageName}.tif',
                   f'/Volumes/SSD_2/Malaysia/AMSR2/DSC/01_tif/SM_C1/extent/AMSR2_D_SM_20120703_bit_extentafter_{PageName}.tif',
                   f'/Volumes/SSD_2/Malaysia/AMSR2/DSC/01_tif/VOD_C1/extent/AMSR2_D_VOD_20120703_bit_extentafter_{PageName}.tif']


for sampletif in tqdm(sample_tif_list):
    make01grid(sampletif)
    

