# -*- coding: utf-8 -*-
"""03_mat2tif.ipynb

SMOS.matをまずEASEでtifにして、そのあとWGSにreprojectしてEASEは消す。
全球範囲。

"""

import time
import os, sys
import glob
import scipy.io as sio
import h5py
# import rioxarray as rio
import xarray as xr
import numpy as np
from pyproj import Transformer
from scipy.interpolate import griddata
import pyproj
from osgeo import gdal, gdal_array, ogr,osr
from osgeo.gdalconst import *
import geopandas as gpd
import glob
from tqdm import tqdm


#in_dir_parent = r"F:\MAlaysia\SMOS\SMOSIC\01_download\Received_20230727\Filtered_8k_5k"
in_dir_parent = r"F:\MAlaysia\SMOS\SMOSIC\01_download\Received_20230727\Filtered_8k_5k_2013_SM"
input_file_list =glob.glob(os.path.join(in_dir_parent, "*.mat"))
input_file_name = sys.argv[1]
# input_file_name = 'Filtered_8k_5k_2010_SM.mat'
input_file = os.path.join(in_dir_parent,input_file_name)

AS_DS = sys.argv[2] #"ASC" "DSC"

if "SM" in input_file_name:
  year = os.path.basename(input_file)[:-4].split("_")[-2]
  out_dir = os.path.join(r"F:\MAlaysia\SMOS\SMOSIC\04_ras\SM",AS_DS, year)
else:
  year = os.path.basename(input_file)[:-4].split("_")[-1]
  out_dir = os.path.join(r"F:\MAlaysia\SMOS\SMOSIC\04_ras\VOD",AS_DS, year)


# Gridding to latlon(SMAP): https://gist.github.com/KMarkert/16902163c52e587436e473587b587f52
# https://stackoverflow.com/questions/45972790/how-to-fix-the-reprojection-from-ease-2-grid-product-smap-to-geographic-coordina
# https://stackoverflow.com/questions/45972790/how-to-fix-the-reprojection-from-ease-2-grid-product-smap-to-geographic-coordina
# できた！！！

h5 = h5py.File(input_file)
if AS_DS == "ASC":
  data = h5.get("filtered_SM_ASC")[:,:,:] #array (365, 1388, 584) row should be 584 #"filtered_SM_DESC"
if AS_DS == "DSC":
  data = h5.get("filtered_SM_DESC")[:,:,:]

projOrigin = (-17367530.44,7307375.92) # x,y
projDims = (1388,584) # cols, rows
projRes = (25025.26, 25025.26) # xResolution, yResolution

col = np.arange(0,projDims[0],1).astype(np.int16)
row = np.arange(0,projDims[1],1).astype(np.int16)[::-1]

xx,yy = np.meshgrid(col,row) #row, colのインデックス

#yy.shape #(584, 1388) xx.shapeも同じ

projX = np.arange(projOrigin[0],projOrigin[0]+((projDims[0])*projRes[0]),projRes[0]).astype(np.float32)
projY = np.arange(projOrigin[1]-((projDims[1])*projRes[1]),projOrigin[1],projRes[1]).astype(np.float32)

xxProj,yyProj = np.meshgrid(projX,projY)

# using proj to reproject EASE coordinates to lat/lon
transformer = Transformer.from_crs("epsg:6933", "epsg:4326",always_xy=True)
projLon,projLat = transformer.transform(xxProj,yyProj)

# make flatten data
yidx = np.ravel(yy) # row info to series
xidx = np.ravel(xx) # col info to series

xcoord = projLon[0,:].astype(np.float32)
ycoord = projLat[:,0].astype(np.float32)

np_lat = np.array(ycoord)
np_lon = np.array(xcoord)

num_cols = float(1388)
num_rows = float(584)

xmin = np_lon.min()
xmax = np_lon.max()
ymin = np_lat.min()
ymax = np_lat.max()
# xres = (xmax - xmin) / num_cols #0.25917915102384276
# yres = (ymax - ymin) / num_rows #0.286215168156036

nrows, ncols = 584, 1388
xres = (xmax - xmin) / float(ncols)
yres = (ymax - ymin) / float(nrows)

# geotransform = (-17367530.44, 25000, 0, 7307375.92, 0, -25000.0)
geotransform = (-17367530.44, 25025.26, 0, 7307375.92, 0, -25025.26)

for i in tqdm(range(365)):
    i_str = str(i+1).zfill(3)
    sm_reshape = data[i].T # 転置する. [100] is now for 1 days #(584, 1388)
    sm = np.ravel(sm_reshape)
    
    # griddataはscipyのinterpolation
    smGridded = griddata(np.stack([yidx,xidx],axis=-1), sm, (yy,xx), fill_value=np.nan, method='linear') #np.stack(-1)でx,yを横に並べる
    np_data = np.array(smGridded)
   
    filename = os.path.basename(input_file)[:-4] + i_str
    
    os.makedirs(out_dir, exist_ok=True)
    dataFileOutput = os.path.join(out_dir,f'{filename}.tif')
    output_raster = gdal.GetDriverByName('GTiff').Create(dataFileOutput, ncols, nrows, 1, gdal.GDT_Float32)  # Open the file
    output_raster.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(6933)
    
    output_raster.SetProjection(srs.ExportToWkt())
    output_raster.GetRasterBand(1).WriteArray(np_data)  # Writes my array to the raster
    
    del output_raster
    
    
    """outputしたtifをreprojectする"""
    #time.sleep(3)
    
    input_raster_path = dataFileOutput
    input_ds = gdal.Open(input_raster_path)
    output_projection = 'EPSG:4326'
    output_raster_path = input_raster_path.replace(".tif","_4326.tif")
    
    gdal.Warp(output_raster_path, input_ds, format='GTiff', dstSRS=output_projection)
    # Close the input dataset
    input_ds = None
    
    os.remove(input_raster_path) #EASE
