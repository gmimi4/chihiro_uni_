# -*- coding: utf-8 -*-
"""
# 0.1 degree gridの中から一定以上のpalm面積があるインデックスを取得する
# Do this code after whole area palm points are created
"""

import numpy as np
import pandas as pd
import os,sys
import rasterio
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
# from osgeo import gdal, ogr
# import matplotlib.pyplot as plt

sample_tif = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_01\_mosaic\mosaic_Eb_importance_2002-2012.tif"
palm_points = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index\palm_index_points_whole.shp"
out_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index"

src = rasterio.open(sample_tif)
src_arr = src.read(1)
h = src_arr.shape[0]
w = src_arr.shape[1]
src_arr_tmp = np.arange(h*w).reshape(h, w).astype("int32")


""" #0.1 degree gridポリゴンをつくる  # This process is done only one time
     use already existed one"""
     
outgrid_name =  f"grid_01degree_{h}_{w}.shp"
if not os.path.isfile(out_dir + os.sep +outgrid_name):
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
else:
    gpd_polygonized_raster = gpd.read_file(out_dir + os.sep +outgrid_name)



""" # ポイントがある位置の解析ラスターのindexを拾う """
gdf_points = gpd.read_file(palm_points)
points = gdf_points.geometry.tolist()

## pick values ###
locations = [(p.xy[0][0],p.xy[1][0]) for p in points]
pixel_indices = [src.index(x, y) for x, y in locations] #sample rasのsrc

#2Dインデックスを1Dに変換したときの1Dインデックスを拾う
oned_indx_list = []
num_cols = w
for rc in pixel_indices:
    row, col = rc[0], rc[1]
    index_1d = row * num_cols + col
    oned_indx_list.append(index_1d)
    
#txtで出力
out_txt = os.path.join(out_dir,f"palm_index_shape_{h}_{w}.txt")
with open(out_txt, "w") as file:
    for idx in oned_indx_list:
        file.write("%s\n" % idx)


src.close()
