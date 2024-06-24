# -*- coding: utf-8 -*-
"""
0.1 degree gridの中から一定以上のpalm面積があるインデックスを取得する
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

PageName = 'A1'
# sample_tif = f'/Volumes/SSD_2/Malaysia/02_Timeseries/CPA_CPR/2_out_ras/p_01/{PageName}_Eb_importance_2002-2012.tif'
sample_tif = rf'D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_01\{PageName}_Eb_importance_2002-2012.tif'
peat_ras = r'D:\Malaysia\SoilMap\Peat\peat_merge_bool.tif'
out_dir = os.path.dirname(peat_ras) + os.sep + "peat_index"

src = rasterio.open(sample_tif)
src_arr = src.read(1)
h = src_arr.shape[0]
w = src_arr.shape[1]
src_arr_tmp = np.arange(h*w).reshape(h, w).astype("int16")

""" #0.1 degree gridポリゴンをつくる 
     use already existed one"""
grid_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index"
outgrid_name =  f"grid_01degree_{PageName}.shp"
if not os.path.isfile(grid_dir + os.sep +outgrid_name):
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
    gpd_polygonized_raster.to_file(os.path.join(grid_dir, outgrid_name), crs="epsg:4326")
else:
    gpd_polygonized_raster = gpd.read_file(grid_dir + os.sep +outgrid_name)


""" # gridポリゴンごとにpeatを含む面積を拾う。 """
### zonalstatでカウントを拾う
palmras_name = os.path.basename(peat_ras)[:-4]

with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
    with rasterio.open(peat_ras) as src_palm:
        meta_palm = src_palm.meta
        kwds = src_palm.profile
        arr_palm = src_palm.read(1).astype('float16')
        arr_palm[arr_palm==0] = np.nan
        affine= src.transform
        # meta_palm.update({"dtype":'float16',"nodata":np.nan})
        kwds['dtype'] = 'float32' #float16はダメだった
        kwds['nodata']= np.nan
        palmras_nan_name = os.path.join(out_dir,f"{palmras_name}_nan_{PageName}.tif")
        with rasterio.open(palmras_nan_name, "w", **kwds) as dst:
            dst.write(arr_palm,1)


stat_dic = zonal_stats(os.path.join(grid_dir, outgrid_name), palmras_nan_name, 
                       stats="count") #affine=affine

# area_ratioが一定以上のグリッドポリゴンをidxで拾う(ここまでok)
count_list = []
for i,d in enumerate(stat_dic):
    for k,co in d.items():
        area_ratio = co*(500**2) / (100*10**6) #peat 1ピクセル500**2m2, 1グリッド 100*10**6m2
        if area_ratio>0.3:
            count_list.append(i)

# indexでグリッドポリゴン抽出
target_grid = gpd_polygonized_raster.iloc[count_list]
# pointにする
target_point = target_grid.copy()
target_point["geometry"] = target_point["geometry"].representative_point()

# パームを一定面積以上含むpointをExport
# target_point.to_file(os.path.join(out_dir,"palm_area_points.shp"))


""" # ポイントがある位置の解析ラスター(=sample tif =importance tif)のindexを拾う """
## pick values ###
locations = target_point.geometry.tolist()
locations = [(p.xy[0][0],p.xy[1][0]) for p in locations]
pixel_indices = [src.index(x, y) for x, y in locations] #sample rasのsrc

#2Dインデックスを1Dに変換したときの1Dインデックスを拾う
oned_indx_list = []
num_cols = w
for rc in pixel_indices:
    row, col = rc[0], rc[1]
    index_1d = row * num_cols + col
    oned_indx_list.append(index_1d)
    
#txtで出力
out_txt = os.path.join(out_dir,f"peat_index_shape_{PageName}.txt")
with open(out_txt, "w") as file:
    for idx in oned_indx_list:
        file.write("%s\n" % idx)
