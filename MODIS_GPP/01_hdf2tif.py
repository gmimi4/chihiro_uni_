# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:11:57 2024
https://gis.stackexchange.com/questions/194533/convert-modis-hdf-file-in-sinusoidal-projection-into-geotiff-using-python-gdal
# さくっとできた
@author: chihiro
"""

import os
from osgeo import gdal
import numpy as np
import glob
from tqdm import tqdm
# import rasterio
# from matplotlib import pyplot

in_dir = r"D:\Malaysia\MODIS_GPP\01_download_Indonesia"
out_dir = r"D:\Malaysia\MODIS_GPP\02_tif_Indonesia"

# in_file = r"D:\Malaysia\MODIS_GPP\01_download_Indonesia\MOD17A2HGF.A2000001.h27v08.061.2020125212306.hdf" # raw MODIS HDF in sinusoid projection
# out_file = r"D:\Malaysia\MODIS_GPP\02_tif_Indonesia\h27v08.061.2020125212306_int.tif"
# QC_dir = r"D:\Malaysia\MODIS_GPP\02_tif_Indonesia\QC" #GFなので不要
# out_file_qc = QC_dir + os.sep + "h27v08.061.2020125212306_QC.tif"

hdfs = glob.glob(in_dir + os.sep + "*.hdf")

for in_file in tqdm(hdfs):
    use_filename = os.path.basename(in_file)[20:].replace(".hdf","")
    out_file = out_dir + os.sep + f"{use_filename}_int.tif"
    """ #いちど raw valのtif出力 """
    # QC不要
    dataset = gdal.Open(in_file,gdal.GA_ReadOnly)
    subdataset =  gdal.Open(dataset.GetSubDatasets()[0][0], gdal.GA_ReadOnly)
    
    # gdalwarp
    kwargs = {'format': 'GTiff', 'dstSRS': 'EPSG:4326'}
    ds = gdal.Warp(destNameOrDestDS=out_file,srcDSOrSrcDSTab=subdataset, **kwargs)
    del ds
    
    
    """ #valid valuefloatに変換 """
    # Read data in arrays
    dataset = gdal.Open(out_file, gdal.GA_ReadOnly)
    # dataset = gdal.Open(out_file)
    # band = dataset.GetRasterBand(1)
    # test = band.ReadAsArray()
    gpp_get = dataset.ReadAsArray()
    gpp_get_32 = gpp_get.astype('float32')
    
    gpp_get_32[np.where ((gpp_get_32==float(32767)) |(gpp_get_32==float(32766))|
                         (gpp_get_32==float(32765))|(gpp_get_32==float(32764))|
                         (gpp_get_32==float(32763))|(gpp_get_32==float(32762))|
                         (gpp_get_32==float(32761)))
               ] = np.nan #filled valueはnanにする
    
    gpp_real = gpp_get_32 * 0.0001
    
    geoT = dataset.GetGeoTransform()
    driver = gdal.GetDriverByName('GTiff')
    pixelwidth_x, pixelwidth_y = geoT[1], geoT[5]
    height, width = gpp_real.shape[0], gpp_real.shape[1] 
    
    out_file_fin = out_dir + os.sep + f"{use_filename}.tif"
    dst = driver.Create(out_file_fin, width, height, 1, gdal.GDT_Float32)
    # とらりもん output.SetGeoTransform((ul_x, h_res, 0, ul_y, 0, v_res))
    dst.SetGeoTransform(geoT)
    dst.SetProjection(dataset.GetProjection())
    
    output_band = dst.GetRasterBand(1)
    output_band.WriteArray(gpp_real)
    
    dst = None
    output_band = None
    dataset = None
    
    os.remove(out_file)

#check
# src = rasterio.open(out_file)
# arr = src.read(1)
# pyplot.imshow(arr)