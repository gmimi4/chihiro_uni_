# -*- coding: utf-8 -*-
"""
変数のvalueを列時間、行ピクセルインデックスの順で並べる

@author: chihiro

"""
#####
#8 days meanができた後
#####

import os, sys
import glob
import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
import pathlib
import datetime
# from rasterio.enums import Resampling
from shapely.geometry import Polygon
from rasterio.mask import mask
from tqdm import tqdm
from rasterio.crs import CRS

with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
    rio_crs = CRS.from_epsg(4326)
    proj_crs = CRS.from_user_input(rio_crs)
    
# GPP_dir = r"D:\Malaysia\MODIS_GPP\02_tif\res_01\merge"
GOSIF_dir = "/Volumes/PortableSSD/Malaysia/SIF/GOSIF/02_tif_age_adjusted/res_01"
# KBDI_dir = r"D:\Malaysia\KBDI\1_daily"
# VPD_dir = "/Volumes/PortableSSD/Malaysia/ECMWF/VPD"
# rain_dir = "/Volumes/SSD_2/Malaysia/GPM/01_tif_Affine"
# temp_dir = "/Volumes/PortableSSD/Malaysia/ECMWF/Temperature_2m/02_tif/daily_1950"
# Et_dir = "/Volumes/PortableSSD/MAlaysia/GLEAM/02_tif_v41/Et"
# Eb_dir = "/Volumes/PortableSSD/MAlaysia/GLEAM/02_tif_v41/Eb"
# Et_dir = "/Volumes/PortableSSD/Malaysia/GLEAM/02_tif/Et/res_01"
# Eb_dir = "/Volumes/PortableSSD/Malaysia/GLEAM/02_tif/Eb/res_01"
# SMASC_dir = r"F:\MAlaysia\SMOS\SMOSIC\04_ras\SM\ASC\res_01"
SMDSCE_dir = "/Volumes/PortableSSD 1/Malaysia/AMSRE/01_tif/SM_C/sameday"
# SMDSC2_dir = "/Volumes/SSD_2/Malaysia/AMSR2/DSC/01_tif/SM_C1"
# VODASC_dir = r"F:\MAlaysia\SMOS\SMOSIC\04_ras\VOD\ASC\res_01"
VODDSCE_dir = "/Volumes/PortableSSD 1/Malaysia/AMSRE/01_tif/VOD_C/sameday"
# VODDSC2_dir = "/Volumes/SSD_2/Malaysia/AMSR2/DSC/01_tif/VOD_C1"

# out_dir = "/Volumes/SSD_2/Malaysia/02_Timeseries/CPA_CPR/0_vars_timeseries"
out_dir = "/Volumes/SSD_2/Malaysia/02_Timeseries/CPA_CPR/0_vars_timeseries/until2023"

# Malaysia_land_shape = r"C:\Users\chihiro\Desktop\PhD\Malaysia\AOI\Administration\National_boundary\Malaysia_national_boundary.shp"
Malaysia_land_shape = "/Volumes/PortableSSD/Malaysia/AOI/extent/Malaysia_and_Indonesia_extent_divided.shp"

GOSIF_list = glob.glob(GOSIF_dir+os.sep+"*.tif")
# GPP_list = glob.glob(GPP_dir+"\\*.tif")
# KBDI_list = glob.glob(KBDI_dir+"\\*.tif")
# VPD_list = glob.glob(VPD_dir+os.sep+"*.tif")
# rain_list = glob.glob(rain_dir+os.sep+"*.tif")
# temp_list = glob.glob(temp_dir+os.sep+"*.tif")
# Et_list = glob.glob(Et_dir+os.sep+"*.tif")
# Eb_list = glob.glob(Eb_dir+os.sep+"*.tif")
# SMASC_list = glob.glob(SMASC_dir+"\\*.tif")
SMDSCE_list = glob.glob(SMDSCE_dir+os.sep+"*.tif")
# SMDSC2_list = glob.glob(SMDSC2_dir+os.sep+"*.tif")
# VODASC_list = glob.glob(VODASC_dir+"\\*.tif")
VODDSCE_list = glob.glob(VODDSCE_dir+os.sep+"*.tif")
# VODDSC2_list = glob.glob(VODDSC2_dir+os.sep+"*.tif")



data_kinds_dic = {#"GPP":GPP_list,"KBDI":KBDI_list,
                   # "GOSIF":GOSIF_list,
                   # "rain":rain_list,
                   # "temp":temp_list,
                   #  "Et":Et_list,
                   # "Eb":Eb_list,
                   # "VPD":VPD_list,
                   "SMDSCE":SMDSCE_list,
                   "VODDSCE":VODDSCE_list,
                   # "SMDSC2":SMDSC2_list,
                   # "VODDSC2":VODDSC2_list
                  }

"""
# preparation
(Malaysia shapeに合わせてExtent設定、shape設定）
"""
gdf_land_Malaysia = gpd.read_file(Malaysia_land_shape)

### Set extent #3ピクセルくらいバッファーつける. make dictionary
extent_polygons ={}
for i, row in gdf_land_Malaysia.iterrows():
    poly = row.geometry
    polyname = row.PageName
    buff_land = poly.buffer(0.3)
    # buff_land = gdf_land_Malaysia.buffer(0.3)
    # 以下はboundary polygonを使ったときextent polygonを作る用
    # bounds = buff_land.bounds
    # xmin, ymin, xmax, ymax = bounds["minx"][0],bounds["miny"][0],bounds["maxx"][0],bounds["maxy"][0]
    # polygon_geom = Polygon([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
    # extent_polygon = gpd.GeoDataFrame(geometry=[polygon_geom], crs=proj_crs)
    extent_polygon = gpd.GeoDataFrame(geometry=[buff_land], crs=proj_crs)
    
    extent_polygons[polyname] = extent_polygon


def get_yyyyddmm(tif_path,datatype):
    # tif_path = rain_list[0]
    p_file = pathlib.Path(tif_path)
    filename = p_file.stem

    if datatype == "GOSIF":
        str_past_days = filename.split("_")[1]
        year = int(str_past_days[0:4])
        past_days = int(str_past_days[4:])
        dt1 = datetime.datetime(year=year, month=1, day=1)
        yyyymmdd_use = dt1 + datetime.timedelta(days=past_days-1)
    if datatype == "temp":
        yyyyddmm_str = filename.split("_")[1]
        yyyymmdd_use = datetime.datetime.strptime(yyyyddmm_str, '%Y%m%d')
    if datatype == "rain":
        yyyyddmm_str = filename.split("_")[1]
        yyyymmdd_use = datetime.datetime.strptime(yyyyddmm_str, '%Y%m%d')
    if datatype == "KBDI":
        yyyyddmm_str = filename.split("_")[1]
        yyyymmdd_use = datetime.datetime.strptime(yyyyddmm_str, '%Y%m%d')
    if datatype == "VPD":
        yyyyddmm_str = filename.split("_")[1]
        yyyymmdd_use = datetime.datetime.strptime(yyyyddmm_str, '%Y%m%d')
    if datatype == "GPP":
        str_past_days = filename.split("_")[1][1:]
        year = int(str_past_days[0:4])
        past_days = int(str_past_days[4:])
        dt1 = datetime.datetime(year=year, month=1, day=1)
        yyyymmdd_use = dt1 + datetime.timedelta(days=past_days-1)
    if datatype == "Et" or datatype == "Eb" or datatype == "Ei":
        str_past_days = filename.split("_")[1]
        year = int(str_past_days[0:4])
        past_days = int(str_past_days[4:])
        dt1 = datetime.datetime(year=year, month=1, day=1)
        yyyymmdd_use = dt1 + datetime.timedelta(days=past_days-1)
    if  datatype == "SMASC2" or datatype == "SMDSC2":
        yyyyddmm_str = filename.split("_")[3]
        yyyymmdd_use = datetime.datetime.strptime(yyyyddmm_str, '%Y%m%d')
    if  datatype == "VODASC2" or datatype == "VODDSC2":
        yyyyddmm_str = filename.split("_")[3]
        yyyymmdd_use = datetime.datetime.strptime(yyyyddmm_str, '%Y%m%d')
    if  datatype == "SMASCE" or datatype == "SMDSCE":
        yyyyddmm_str = filename.split("_")[4]
        yyyymmdd_use = datetime.datetime.strptime(yyyyddmm_str, '%Y%m%d')
    if  datatype == "VODASCE" or datatype == "VODDSCE":
        yyyyddmm_str = filename.split("_")[4]
        yyyymmdd_use = datetime.datetime.strptime(yyyyddmm_str, '%Y%m%d')
    """ #SMOS
    if  datatype == "SMASC" or datatype == "SMDSC":
        str_past_days = filename.split("_")[4][2:]
        year = int(filename.split("_")[3])
        past_days = int(str_past_days)
        dt1 = datetime.datetime(year=year, month=1, day=1)
        yyyymmdd_use = dt1 + datetime.timedelta(days=past_days-1)
    if  datatype == "VODASC" or datatype == "VODDSC":
        str_past_days = filename.split("_")[3]
        year = int(str_past_days[0:4])
        past_days = int(str_past_days[4:])
        dt1 = datetime.datetime(year=year, month=1, day=1)
        yyyymmdd_use = dt1 + datetime.timedelta(days=past_days-1)
    """
    return yyyymmdd_use

    
"""
# 処理
"""
## First, obtain GOSIF crop shape
# page = "A4"
# extent_poly= extent_polygons[page] #this should be done in advance...
def extent_shape(extent_poly):
    tiflist = GOSIF_list
    tif = tiflist[-1]
    with rasterio.open(tif) as src:
        data = src.read()
        ### ---------------------------------
        #crop to extent
        ### ---------------------------------
        cropped_ras, raster_transform = mask(src, extent_poly.geometry, crop=True)
        # meta=src.meta.copy()
        # meta.update({"transform":raster_transform,"width":shape_use[1],"height":shape_use[0],
        #               "crs":proj_crs})
        if len(cropped_ras.shape) ==3:
            cropped_ras = cropped_ras[0]
        else:
            pass
        sif_shape = cropped_ras.shape
        
        # ### check
        # meta=src.meta.copy()
        # meta.update({"transform":raster_transform,"width":cropped_ras.shape[1],"height":cropped_ras.shape[0],
        #               "crs":proj_crs})
        # extent_dir = os.path.dirname(tif)+os.sep +"extent"
        # os.makedirs(extent_dir,exist_ok=True)        
        # path_to_output = extent_dir + os.sep +f"{os.path.basename(tif[:-4])}_extent_{page}.tif"
        # with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
        #     with rasterio.open(path_to_output,"w",**meta) as dst:
        #         dst.write(cropped_ras,1)
        
        return sif_shape
    

## Make arr shape same as gosif's shape
def set_shape(arr, shape_use):
    # arr = cropped_ras
    height_diff = arr.shape[0] - shape_use[0]
    width_diff = arr.shape[1] - shape_use[1]
    
    ## drop out the bottom rowa and right col ## 2 pixel以上はずれないと仮定
    # height
    if height_diff ==1:
        arr_cut = np.delete(arr, 0, 0) #drop top rows
    elif height_diff ==2:
        arr_cut = np.delete(arr, [0,-1], 0) #drop top and botom rows
    else:
        arr_cut = arr
    
    # width
    if width_diff ==1:
        arr_cut = np.delete(arr_cut, -1, 1) #drop top rows
    elif width_diff ==2:
        arr_cut = np.delete(arr_cut, [0,-1], 1) #drop top and botom rows
    else:
        arr_cut = arr_cut
    
    return arr_cut
        

## process for each extent
for pagename, extent in extent_polygons.items():
    # pagename,extent = "A1", extent_polygons["A1"]
    
    ## crop other rasters to same as sif.shape 
    sif_shape = extent_shape(extent)
    
    for kind, tiflist in data_kinds_dic.items():
        # kind, tiflist = "GOSIF", data_kinds_dic["GOSIF"]
        # kind, tiflist = "GPP", data_kinds_dic["GPP"]
        # kind, tiflist = "KBDI", data_kinds_dic["KBDI"]
        # kind, tiflist = "rain", data_kinds_dic["rain"]
        # kind, tiflist = "SMDSC2", data_kinds_dic["SMDSC2"]
        # kind, tiflist = "VODDSC2", data_kinds_dic["VODDSC2"]
        # kind, tiflist = "temp", data_kinds_dic["temp"]
        # kind, tiflist = "Et", data_kinds_dic["Et"]
        
        final_outdir = out_dir + os.sep + pagename
        os.makedirs(final_outdir, exist_ok=True)
        outfile = final_outdir + os.sep + f"{pagename}_{kind}_pixels_dates.csv"
        
        if os.path.isfile(outfile):
            continue
        else:
            print(outfile)
            date_flatten_dic = {}
            for tif in tqdm(tiflist):
                # tif = tiflist[1]
                # tif = [t for t in tiflist if "20120703" in t][0]
                filedate = get_yyyyddmm(tif, kind)
                # Read the raster data
                src = rasterio.open(tif)
                data = src.read()
        
                ### ---------------------------------
                #crop to extent
                ### ---------------------------------
                cropped_ras, raster_transform = mask(src, extent.geometry, crop=True)

                if len(cropped_ras.shape) ==3:
                    cropped_ras = cropped_ras[0]
                else:
                    pass
                
                ### TemperatureとKBDIがたぶんずれててshape違うので臨時で修正
                remove_arr = set_shape(cropped_ras, shape_use = sif_shape)
                cropped_ras = remove_arr
                    
                # if cropped_ras.shape == (72, 204): #事前に確認しないといけない #Malay(72, 204) #GOSIF:(210, 129)
                # # if kind == "temp" or kind =="KBDI" or kind =="SMDSC" or kind =="VODDSC":
                #     remove_arr = cropped_ras[:,:-1]
                #     cropped_ras = remove_arr
                # else:
                #     pass
                
                # ### check
                # meta=src.meta.copy()
                # meta.update({"transform":raster_transform,"width":cropped_ras.shape[1],"height":cropped_ras.shape[0],
                #               "crs":proj_crs})
                # extent_dir = os.path.dirname(tif)+"\\extent"
                # os.makedirs(extent_dir,exist_ok=True)        
                # path_to_output = extent_dir +f"\\{os.path.basename(tif[:-4])}_extent.tif"
                # with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
                #     with rasterio.open(path_to_output,"w",**meta) as dst:
                #         dst.write(cropped_ras,1)
                
                ### ---------------------------------
                #flatten and interelation
                ### ---------------------------------
                cropped_flatten = np.ravel(cropped_ras) #縦になってる(22321,)
                date_flatten_dic[filedate] = cropped_flatten
                
                src.close()
                
            ### to dataframe
            print(cropped_ras.shape)
            df = pd.DataFrame(date_flatten_dic)
            # to csv
            final_outdir = out_dir + os.sep + pagename
            os.makedirs(final_outdir, exist_ok=True)
            outfile = final_outdir + os.sep + f"{pagename}_{kind}_pixels_dates.csv"
            df.to_csv(outfile)
        
       
    
        





