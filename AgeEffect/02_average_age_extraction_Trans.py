# -*- coding: utf-8 -*-
"""
Global Palm area and year map
https://doi.org/10.5194/essd-2024-157
extract grids which contain a certain year as majority in a grid
@author: chihiro
"""

import os
import geopandas as gpd
import pyogrio as pg
# from rasterstats import zonal_stats
from tqdm import tqdm
import pandas as pd
# from collections import Counter

grid_shp = r"F:\MAlaysia\GLEAM\02_tif_v41\Et\_grid\Et_2000001_grid.shp"
age_tif = r"F:\MAlaysia\AOI\High_resolution_global_industrial_and_smallholder_oil_palm_map_for_2019\GlobalOilPalm_OP-YoP\Malaysia_Indonesia\GlobalOilPalm_OP-YoP_mosaic100m.tif"
age_shp = r"F:\MAlaysia\AOI\High_resolution_global_industrial_and_smallholder_oil_palm_map_for_2019\GlobalOilPalm_OP-YoP\Malaysia_Indonesia\shp\GlobalOilPalm_OP-YoP_mosaic100m.shp"
out_dir = r"F:\MAlaysia\AgeEffect\01_test_points"

# grid_shp = "/Volumes/Samsung_X5/AMSR_retrieval/00_rasvals/_grid/AMSRE_20020619_D.6.9_crop_grid_blocks.shp"
# age_tif = "/Volumes/PortableSSD/Malaysia/AOI/High_resolution_global_industrial_and_smallholder_oil_palm_map_for_2019/GlobalOilPalm_OP-YoP/Malaysia_Indonesia/GlobalOilPalm_OP-YoP_mosaic100m.tif"
# age_shp = "/Volumes/PortableSSD/Malaysia/AOI/High_resolution_global_industrial_and_smallholder_oil_palm_map_for_2019/GlobalOilPalm_OP-YoP/Malaysia_Indonesia/shp/GlobalOilPalm_OP-YoP_mosaic100m.shp"
# out_dir = "/Volumes/Samsung_X5/AMSR_retrieval/99_palmage"

gdf_grid = gpd.read_file(grid_shp) #tif's grid
gdf_palm = pg.read_dataframe(age_shp)

# # prepare palm polygon to be used
# gdf_palm_subset = gdf_palm.query('gridcode <= 2002 & gridcode >= 2000') #2002年より前にreplantされたエリアを抽出したい
## 時間かかるのでやめる
# gdf_palm["tmp"] = 1
# gdf_palm_diss = gdf_palm.dissolve(by="tmp")
 
area_grid = round(gdf_grid.loc[0,:].geometry.area, 4) #area of a grid #0.025 degree**2

    
""" # obtain average year"""
# USe all age palm tif
print(len(gdf_grid))
age_grid_dic = {}
for i,row in tqdm(gdf_grid.iterrows()):
    # i=52
    # row = gdf_grid.loc[i,:]
    grid = row.geometry
    gdf_subgrid = gpd.GeoDataFrame({"geometry":[grid]},crs="epsg:4326")
    # clip by sub grid
    clipped = gpd.clip(gdf=gdf_palm, mask=gdf_subgrid)
    # calculate average year by area
    if len(clipped)>0:
        clipped["area"] = clipped.geometry.area
        df_clipped = clipped.drop(columns="geometry")
        df_clipped = df_clipped.groupby("gridcode").sum()
        df_clipped = df_clipped.reset_index()
        all_area = df_clipped["area"].sum()
        df_clipped["rate"] = df_clipped["area"]/all_area
        df_clipped["year_corr"] = df_clipped["rate"]*df_clipped["gridcode"]
        fin_year = df_clipped["year_corr"].sum()
        fin_year = int(fin_year)
        
        age_grid_dic[i] = [fin_year]
    else:
        age_grid_dic[i] = [0]
        continue
        

df_meanyear = pd.DataFrame.from_dict(age_grid_dic).T
df_meanyear.columns=["avage"]
df_meanyear = df_meanyear.reset_index()

gdf_grid_majo = pd.concat([gdf_grid, df_meanyear], axis=1)

outfile = out_dir + os.sep + os.path.basename(grid_shp)[:-4] + "_meanAge.shp"
gdf_grid_majo.to_file(outfile)


