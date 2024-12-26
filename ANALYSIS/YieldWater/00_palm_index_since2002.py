# -*- coding: utf-8 -*-
"""
Extract palm index planted before 2002.
Select 0.1 grid interscting with mean age 0.05 grids
"""
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import glob
import numpy as np
import statistics


shp_01grid_palm2002 = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index\grid_01degree_210_496_palm2002.shp"
out_dir = os.path.dirname(shp_01grid_palm2002)

""" set gdf """
## GOSIF mean age grid already limited to palm area
gdf_grid = gpd.read_file(shp_01grid_palm2002)


""" # select before 2002 grids"""
### Arcでやった
# """ # select before 2002 grids"""
# gdf_age_2002 = gdf_age[gdf_age.avage<=2002]
# ## extract 0.1 grids by intersect
# results=[]
# for i, row in tqdm(gdf_grid.iterrows()):
#     if row.geometry.contains(gdf_age_2002.geometry.any()):
#         results.append(row)        

grid_target_index= [int(i) for i in gdf_grid.raster_val]


""" export to txt """
out_dir = os.path.dirname(shp_01grid_palm2002)
out_filename = os.path.basename(shp_01grid_palm2002)[:-4]
out_txt = os.path.join(out_dir,f"{out_filename}.txt")
with open(out_txt, "w") as file:
    for idx in grid_target_index:
        file.write("%s\n" % idx)


    



