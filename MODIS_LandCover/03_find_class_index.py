# -*- coding: utf-8 -*-
"""
0.1 degree gridの中からmaximum zonal statする
"""

import numpy as np
import os, sys
import rasterio
import geopandas as gpd
from rasterio.features import shapes
# from rasterstats import zonal_stats
from rasterio.mask import mask
from collections import Counter
from shapely.geometry import box
from tqdm import tqdm

PageName = 'A4'
# sample_tif = f'/Volumes/SSD_2/Malaysia/02_Timeseries/CPA_CPR/2_out_ras/p_01/{PageName}_Eb_importance_2002-2012.tif'
sample_tif = rf"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_01\{PageName}_Eb_importance_2002-2012.tif"

ras_path = r'F:\MAlaysia\MODIS_IGBP\MODIS_IGBP_mosaic_reclass.tif'
out_dir = r'F:\MAlaysia\MODIS_IGBP\grid_index'
tmp_dir = os.path.join(out_dir,"tmp")
os.makedirs(tmp_dir,exist_ok=True)

src = rasterio.open(sample_tif)
src_arr = src.read(1)
h = src_arr.shape[0]
w = src_arr.shape[1]
src_arr_tmp = np.arange(h*w).reshape(h, w).astype("int16")

""" #0.1 degree gridポリゴンをつくる 
     use already existed one"""
     
outgrid_name =  rf"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index\grid_01degree_{PageName}.shp"
if not os.path.isfile(outgrid_name):
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


""" # まず不要なデータをnanにする """
ras_name = os.path.basename(ras_path)[:-4]
with rasterio.open(ras_path) as src_palm:
        meta_palm = src_palm.meta
        kwds = src_palm.profile
        arr_palm = src_palm.read(1).astype('float16')
        # test = np.ravel(arr_palm).tolist()
        arr_palm[arr_palm==0] = np.nan
        arr_palm[arr_palm==99] = np.nan #other land
        affine= src.transform
        # meta_palm.update({"dtype":'float16',"nodata":np.nan})
        kwds['dtype'] = 'float32' #float16はダメだった
        kwds['nodata']= np.nan
        palmras_nan_name = os.path.join(tmp_dir,f"{ras_name}_nan_{PageName}.tif")
        
        with rasterio.open(palmras_nan_name, "w", **kwds) as dst:
            dst.write(arr_palm,1)
  
            
""" # majority valueをgridポリに入れていく # zonal_statsはmajority機能ないぽい """  
gdf_grid = gpd.read_file(outgrid_name)

src = rasterio.open(palmras_nan_name)

majority_values = []
for poly in tqdm(gdf_grid.geometry):
    # Create a bounding box around the polygon
    minx, miny, maxx, maxy = poly.bounds
    bbox = box(minx, miny, maxx, maxy)

    # Mask the raster data with the polygon
    try:
        out_image, out_transform = mask(src, [poly], crop=True)
    except:
        majority_values.append(np.nan)
        continue

    # Flatten the array and filter out the nodata values
    flat_data = out_image[0].flatten()
    flat_data = flat_data[flat_data != src.nodata]

    # Calculate the majority value
    if len(flat_data) > 0:
        unique, counts = np.unique(flat_data, return_counts=True)
        majority_value = unique[np.argmax(counts)]
    else:
        majority_value = np.nan

    majority_values.append(majority_value)

# Add the majority values to the GeoDataFrame
gdf_grid['majority_value'] = majority_values
src.close()


""" # class別にインデックス出力 """
df_grid = gdf_grid.loc[:,"majority_value"]
index_forest = df_grid.index[df_grid==40]
index_shrub = df_grid.index[df_grid==30]
index_crop = df_grid.index[df_grid==20]

index_dic ={"forest":index_forest,"shrub":index_shrub,"crop":index_crop}

#txtで出力
for land, oned_indx_list in index_dic.items():
    out_txt = os.path.join(out_dir,f"{land}_index_{PageName}.txt")
    with open(out_txt, "w") as file:
        for idx in oned_indx_list:
            file.write("%s\n" % idx)
