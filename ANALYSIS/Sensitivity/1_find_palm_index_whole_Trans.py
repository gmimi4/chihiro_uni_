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
from rasterstats import zonal_stats

#---------------------------------------
""" # create palm points """
#---------------------------------------
sample_tif = r"F:\MAlaysia\GLEAM\02_tif_v41\Et\Et_2000001.tif"
shp_grid = r"F:\MAlaysia\GLEAM\02_tif_v41\Et\_grid\Et_2000001_grid.shp" #grid poly of tif 
palm_ras = r'F:\MAlaysia\AOI\High_resolution_global_industrial_and_smallholder_oil_palm_map_for_2019\GlobalOilPalm_OP-YoP\Malaysia_Indonesia\GlobalOilPalm_OP-YoP_mosaic100m.tif'
out_dir = r'D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index_Trans'

src = rasterio.open(sample_tif)
src_arr = src.read(1)
h = src_arr.shape[0]
w = src_arr.shape[1]
src_arr_tmp = np.arange(h*w).reshape(h, w).astype("int32")

""" # gridポリゴンごとにpalmを含む面積を拾う。 """
### zonalstatでカウントを拾う
palmras_name = os.path.basename(palm_ras)[:-4]

## Clean palm raster
with rasterio.open(palm_ras) as src_palm:
    meta_palm = src_palm.meta
    kwds = src_palm.profile
    arr_palm = src_palm.read(1).astype('float16')
    arr_palm[arr_palm==0] = np.nan
    arr_palm[arr_palm>2030] = np.nan #inf exist
    affine= src.transform
    # meta_palm.update({"dtype":'float16',"nodata":np.nan})
    kwds['dtype'] = 'float32' #float16はダメだった
    kwds['nodata']= np.nan
    palmras_nan_name = os.path.join(out_dir,f"{palmras_name}_nan.tif")
    with rasterio.open(palmras_nan_name, "w", **kwds) as dst:
        dst.write(arr_palm,1)

#いっかい出さないとだめぽい
stat_dic = zonal_stats(shp_grid, palmras_nan_name, stats="count") #affine=affine

# area_ratioが一定以上のグリッドポリゴンをidxで拾う
count_list = []
for i,d in enumerate(stat_dic):
    for k,co in d.items():
        area_ratio = co*(100**2) / (100*10**6) #palm 1ピクセル100**2m2, 1グリッド 100*10**6m2
        if area_ratio>0.1: #0.3
            count_list.append(i)

# indexでグリッドポリゴン抽出
target_grid = gpd_polygonized_raster.iloc[count_list]
# pointにする
target_point = target_grid.copy()
target_point["geometry"] = target_point["geometry"].representative_point()

# パームを一定面積以上含むpointをExport
tmp_dir = out_dir + os.sep + "_pointsA"
os.makedirs(tmp_dir,exist_ok=True)
target_point.to_file(os.path.join(tmp_dir,f"palm_area_points_{PageName}.shp"))


#--------------------------------------------

sample_tif = r"F:\MAlaysia\GLEAM\02_tif_v41\Et\Et_2000001.tif"
palm_points = "/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/0_palm_index_EVI/_pointsA/palm_area_points_whole.shp"
out_dir = "/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/0_palm_index_EVI"

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
