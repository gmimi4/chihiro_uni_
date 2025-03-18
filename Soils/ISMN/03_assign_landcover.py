# -*- coding: utf-8 -*-
"""
Assign landcover from IGBP to ISMN stations (vegetation)
"""
import os
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from collections import Counter
from tqdm import tqdm
os.chdir(r"C:\Users\chihiro\Desktop\Python\MODIS_LandCover")
import _02_extract_classes

""" #Class
10:grassland
20:cropland
30:shrub
41:forest 常緑
42:forest 落葉
43:forest 混合
50:water
60:barren
70:snow
99:others
"""
classes=[10,20,30,41,42,43,50,60,70,99]

csv_meta = r"F:\MAlaysia\Soil\ISMN\00_download\ALL_Vegetation\python_metadata\Data_separate_files_header_20020101_20231231_10469_QyDr_20250215.csv"
vegesite = r"F:\MAlaysia\Soil\ISMN\02_shp\vegesite.shp"
amsr_grid = r"F:\MAlaysia\Soil\AMSR_retrieval\00_prepartation\amsr_grid_vegepoints.shp"
IGBP_dir = r"F:\MAlaysia\MODIS_IGBP\ISMN_grids"
palm_grid = r"F:\MAlaysia\AOI\High_resolution_global_industrial_and_smallholder_oil_palm_map_for_2019\Grid_OilPalm2016-2021\Grid_OilPalm2016-2021.shp"
palm_dir = r"F:\MAlaysia\AOI\High_resolution_global_industrial_and_smallholder_oil_palm_map_for_2019\GlobalOilPalm_OP-YoP"
out_dir = r"F:\MAlaysia\Soil\ISMN\03_landcover"

df_csv = pd.read_csv(csv_meta, header=[0,1]) #ISMN vegesites
df_csv_use = df_csv.loc[:,[('latitude','val'), ('longitude','val'), ('network','val'), ('station','val'),
                           ('timerange_from','val'), ('timerange_to','val')]]

gdf_amsr_grid = gpd.read_file(amsr_grid)
gdf_palm_grid = gpd.read_file(palm_grid)
gdf_vegesite = gpd.read_file(vegesite)


igbp_tifs = glob.glob(IGBP_dir + os.sep + "*.tif")
palm_tifs = glob.glob(palm_dir + os.sep + "*.tif")



vegesite_with_palm = []
results = []
for i, metadt in tqdm(df_csv_use.iterrows(), total=len(df_csv_use)):
    network = metadt[("network",	"val")]
    station = metadt[("station",	"val")]
    lat, lon = metadt[('latitude','val')], metadt[('longitude','val')]
    point_station = Point(lon, lat)
    amr_targrid = gdf_amsr_grid[gdf_amsr_grid.geometry.intersects(point_station)]
    amsr_grid_code = amr_targrid["gridcode"].values[0]
        
    # -------------------------------------------------
    """#find grid where station located"""
    # -------------------------------------------------
    """ #collect target IGBP tif"""
    tar_year = metadt.loc[("timerange_to","val")][0:4] #'2016' str
    igbp_file = [t for t in igbp_tifs if os.path.basename(t)[:-8] == f"IGBP_{amsr_grid_code}_{network}_to{tar_year}"][0]
    
    """ #reclassify array"""
    igbp_arr =_02_extract_classes.main(igbp_file)
    igbp_arr_1d = np.ravel(igbp_arr)
    igbp_arr_1d_valid = igbp_arr_1d[igbp_arr_1d != 0]
    
    ### calc ratio
    igbp_class, counts = np.unique(igbp_arr_1d_valid, return_counts=True)
    igbp_counts = dict(zip(igbp_class, counts)) #dictionary
    
    df_default = pd.DataFrame([[0]* len(classes)], columns=classes)
    
    for cl, nums in igbp_counts.items():
        df_default.loc[0,cl] = nums # input nums of class
    df_ratio = df_default / len(igbp_arr_1d_valid) # convert to ratio
    
    df_ratio["network"] = network
    df_ratio["station"] = station
    df_ratio["gridcode"] = amsr_grid_code
    
    
    # -------------------------------------------------
    """#extract palm tif where station located""" 
    # -------------------------------------------------
    palm_targrid = gdf_palm_grid[gdf_palm_grid.geometry.intersects(point_station)]
    
    if len(palm_targrid)>0: ## this site overlays with palm tif
        vegesite_with_palm.append(i)
        # i = vegesite_with_palm[0]
        # metadt = df_csv_use.loc[i,:]
        palm_grid_code = int(palm_targrid["ID"].values[0])
        tif_palm = [t for t in palm_tifs if os.path.basename(t)[:-4].split("_")[-1]==str(palm_grid_code)][0]## extract palm tif
        
        """ # clip palm tif by AMSR grid"""
        with rasterio.open(tif_palm) as src:
            meta = src.meta.copy()
            clipped_palm, transform_new = mask(src, amr_targrid.geometry, crop=True)
            meta.update({"height":clipped_palm.shape[1],
                         "width":clipped_palm.shape[2],
                         "transform":transform_new})
            
        """ # test export"""
        tmp_dir = out_dir + os.sep + "_tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        outpalm = tmp_dir + os.sep + f"OP-YoP_amsr{amsr_grid_code}.tif"
        with rasterio.open(outpalm,"w",**meta) as dst:
            dst.write(clipped_palm)
        
        # """ # resample palm 0.005 as similar to IGBP""" #--> not do
        
        """ # calc ratio of palm """
        clipped_palm_validnum = len(clipped_palm[0][clipped_palm[0]!=0]) #num of val not 0
        palm_ratio = clipped_palm_validnum / len(np.ravel(clipped_palm[0]))
    
    else:
        palm_ratio =0
    
    df_ratio["palm"] = palm_ratio
    
    results.append(df_ratio)

    
    
df_concat = pd.concat(results)
df_concat = df_concat.reset_index(drop=True)
df_concat.to_csv(out_dir + os.sep + "stations_landcover.csv")
    
