# -*- coding: utf-8 -*-
"""
# Assign majority k-mean class to regions
"""
import os
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats


shp_region = "/Volumes/SSD_2/Malaysia/Validation/1_Yield_doc/shp/region_slope_fin.shp"
# tiffile = '/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/14_kclustering/mosaic_min_rain_cluster.tif'
tiffile = '/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/13_mins/_mosaic/mosaic_min_rain.tif'
# tiffile = '/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/13_mins/_mosaic/mosaic_minmon_rain.tif'
out_dir = os.path.dirname(tiffile)

stat_str="mean"

stats = zonal_stats(
    shp_region,
    tiffile,
    stats=[stat_str],
    geojson_out=True
)

# Extract the majority values
majority_values = [feature['properties'][stat_str] for feature in stats]

# Add the majority value as a new column in the GeoDataFrame
polygons = gpd.read_file(shp_region)
polygons[stat_str] = majority_values
polygons[stat_str] = polygons[stat_str].fillna(-1).astype('int16')  # Replace NaN with -1 for clarity
# polygons["majority"] =polygons["majority"].astype('int16')


# Save the updated shapefile
output_shapefile = out_dir + os.sep + f"{os.path.basename(tiffile)[:-4]}_{stat_str}.shp"
polygons.to_file(output_shapefile)
 
## Export grouping csv
df_region = polygons[["Name",stat_str]]
df_region = df_region.set_index("Name")
df_region = df_region.sort_values(by=stat_str)
output_file = out_dir + os.sep + f"{os.path.basename(tiffile)[:-4]}_{stat_str}.csv"
df_region.to_csv(output_file)


""" concat df (only for min and k-clustering results)"""
csv_k = '/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/14_kclustering/mosaic_min_rain_cluster_majority.csv'
csv_min = '/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/13_mins/_mosaic/mosaic_min_rain_mean.csv'
out_dir2 = os.path.dirname(csv_k)

df_k = pd.read_csv(csv_k, index_col=0)
df_min = pd.read_csv(csv_min, index_col=0)

df_concat =pd.concat([df_k,df_min],axis=1)
df_concat.to_csv(out_dir2+os.sep +f"{os.path.basename(csv_k)[:-4]}_fin.csv")
