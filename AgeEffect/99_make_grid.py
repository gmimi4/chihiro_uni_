# -*- coding: utf-8 -*-
"""
make grid polygon of AMSR 
"""
import os,sys
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import shape
from rasterio.features import shapes
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
import glob

sample_AMSRE = '/Users/wtakeuchi/Desktop/L2G_tif/20020619_D.6.9.tif'
sample_AMSR2 = '/Users/wtakeuchi/Desktop/L2G2_tif/20120703_D.6.9.tif'
out_dir = '/Users/wtakeuchi/Desktop/AMSR_retrieval/00_rasvals/_grid'

tmp_dir = out_dir + os.sep + '_tmp'
os.makedirs(tmp_dir, exist_ok=True)

# ---------------------------------
""" crop first"""
# ---------------------------------
min_lon = 93
min_lat = -12
max_lon = 142
max_lat = 9

for amsr, rasfile in {"AMSRE":sample_AMSRE, "AMSR2":sample_AMSR2}.items():
    rasname = os.path.basename(rasfile)[:-4]
    with rasterio.open(rasfile) as src:
        bbox = rasterio.transform.from_bounds(min_lon, min_lat, max_lon, max_lat, src.width, src.height)
        bounds = rasterio.warp.transform_bounds({'init': 'EPSG:4326'}, src.crs, min_lon, min_lat, max_lon, max_lat)
        window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], src.transform)
        cropped_data = src.read(window=window)
        cropped_transform = src.window_transform(window)

        output_path = tmp_dir +os.sep + f"{amsr}_{rasname}_crop.tif"
        with rasterio.open(
            output_path,
            "w",
            driver=src.driver,
            height=cropped_data.shape[1],
            width=cropped_data.shape[2],
            count=src.count,
            dtype=src.dtypes[0],
            crs=src.crs,
            transform=cropped_transform,) as dst:
            dst.write(cropped_data)


# ---------------------------------
""" convert raster to polygon grid"""
# ---------------------------------
cropped_rass = glob.glob(tmp_dir+os.sep+"*.tif")

raster_path = cropped_rass[0]
with rasterio.open(raster_path) as src:

    raster_data = src.read(1)
    raster_transform = src.transform
    rows = raster_data.shape[0]
    cols = raster_data.shape[1]
    
    raster_1d = np.ravel(raster_data)
    raster_1d_ser = np.array(range(len(raster_1d))).astype('int32') #input serial values
    raster_reshaped = raster_1d_ser.reshape((rows, cols))
    
    # Convert raster to polygon shapes
    polygons = list(
        shapes(raster_reshaped, transform=raster_transform, connectivity=8)
    )

# Convert the shapes to a GeoDataFrame
geometries = [shape(geom) for geom, value in polygons]
values = [value for geom, value in polygons]

gdf = gpd.GeoDataFrame({"geometry": geometries, "rasval": values}, crs="EPSG:4326")

# Save the GeoDataFrame to a shapefile
output_file = os.path.basename(raster_path)[:-4] +"_grid.shp"
gdf.to_file(out_dir + os.sep + output_file)


