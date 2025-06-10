# -*- coding: utf-8 -*-
"""
"""

import os
import glob
import rasterio
from shapely.geometry import box
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
import geopandas as gpd
import numpy as np
from tqdm import tqdm

# tif_dir = '/Volumes/Samsung_X5/ERA5/Temperature2m/01_tif'
# out_dir = '/Volumes/Samsung_X5/ERA5/Temperature2m/02_resample'
# tmp_dir = '/Volumes/Samsung_X5/ERA5/Temperature2m/02_resample/_tmp'
# tif_dir = '/Users/wtakeuchi/Desktop/L2G2_tif' ## Malaysia and Indonesia extent
# tif_dir = '/Volumes/PortableSSD/Malaysia/MODIS_EVI/01_MOD13A2061_resample/_4326'
tif_dir = "/Volumes/SSD3/Malaysia/AMSRE/01_tif/VOD_C/sameday"
# out_dir = '/Volumes/Samsung_X5/MODIS/EVI/01_resmaple_AMSR'
out_dir = '/Volumes/SSD3/Malaysia/AMSRE/01_tif/VOD_C/sameday_resample_EVI'
# out_dir = tif_dir + os.sep + 'Malaysia_Indonesia'
os.makedirs(out_dir,exist_ok=True)
tmp_dir = out_dir + os.sep + '_tmp'
os.makedirs(tmp_dir,exist_ok=True)
tif_AMSR_sample = '/Volumes/PortableSSD/Malaysia/MODIS_EVI/01_MOD13A2061_resample/_4326_res01_age_adjusted/MODEVI_20000218_4326_res01_adj.tif'

tifs = glob.glob(os.path.join(tif_dir, '*.tif'))


# -----------------------------------
""" # Preparation: Crop AMSR raster to ERA extent"""
# -----------------------------------

# min_lon, min_lat, max_lon, max_lat = -180, -60, 180, 75 #from ERA download py #temp
min_lon, min_lat, max_lon, max_lat = 93, -12, 142, 9 #from ERA download py #Albedo
bbox = box(min_lon, min_lat, max_lon, max_lat)
gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs="EPSG:4326")
gdf.to_file(tmp_dir + os.sep + 'extent_check.shp')


""" # crop AMSR"""
with rasterio.open(tif_AMSR_sample) as src:
    
    left, bottom, right, top = src.bounds
    
    row_min, col_min = src.index(min_lon, max_lat)  # min_lon, max_lat are the top-left corner
    row_max, col_max = src.index(max_lon, min_lat)  # max_lon, min_lat are the bottom-right corner

    window = Window(col_min, row_min, col_max - col_min, row_max - row_min)
    height = row_max - row_min
    width = col_max - col_min
    new_trasnform = src.window_transform(window)
    
    cropped_image = src.read(1, window=window) 
    
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "crs": src.crs,  # CRS (Coordinate Reference System)
        "transform": new_trasnform,  # Update the transform to match the new crop
        'height':height,
        'width':width # maybe should be active
    })
    
    outfile = tmp_dir + os.sep + f'{os.path.basename(tif_AMSR_sample)[:-4]}_crop.tif'
    with rasterio.open(outfile, 'w', **out_meta) as dest:
        dest.write(cropped_image, 1)

        


# -----------------------------------
""" # Resample ERA as AMSR"""
# -----------------------------------

for tif in tqdm(tifs):
    with rasterio.open(tif) as src:
        original_data = src.read(1)  # Read the first band of the original raster
        original_transform = src.transform
        original_crs = src.crs
        # dttype = src.meta['dtype'] #check
    
        resampled_data = np.empty((height, width), dtype=np.float32) #allign with AMSR
        data_min = np.nanmin(resampled_data)
    
        reproject(
            source=original_data,
            destination=resampled_data,
            src_transform=original_transform,
            src_crs=original_crs,
            dst_transform=new_trasnform,
            dst_crs=original_crs,
            resampling=Resampling.nearest  # You can use other resampling methods (e.g., bilinear)
        )
        resampled_data = np.where(resampled_data==data_min, np.nan, resampled_data)

    resampled_raster_path = out_dir + os.sep + f'{os.path.basename(tif)[:-4]}_res.tif'
    with rasterio.open(resampled_raster_path, 'w', driver='GTiff', count=1, dtype='float32',
                       crs=original_crs, transform=new_trasnform, width=width, height=height) as dest:
        dest.write(resampled_data, 1)
        
       

# # -----------------------------------
# """ # Simply crop AMSR themselves"""
# # -----------------------------------
# from rasterio.windows import from_bounds

# for tif in tqdm(tifs):
#     with rasterio.open(tif) as src:

#         window = from_bounds(min_lon, min_lat, max_lon, max_lat, transform=src.transform)
#         transform = src.window_transform(window)
#         data = src.read(window=window)
#         profile = src.profile
#         profile.update({
#             'height': data.shape[1],
#             'width': data.shape[2],
#             'transform': transform,
#             'count': src.count  # number of bands
#         })
    
#         outfile = out_dir + os.sep + f'{os.path.basename(tif)[:-4]}_crop.tif'
#         with rasterio.open(outfile, 'w', **profile) as dst:
#             dst.write(data)





