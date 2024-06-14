# -*- coding: utf-8 -*-
"""
# POC terrace lineとどれくらい近い位置にあるか
# terrace lineからバッファーをいくつか作成してカウントする
# 単純に全体でバッファーをつくって集計する

"""

import os,sys
import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon,MultiPoint,MultiPolygon,LinearRing
from shapely import wkt
from tqdm import tqdm
from statistics import mean
import warnings
from shapely.ops import unary_union
warnings.simplefilter('ignore', FutureWarning)

point_this_study = r"D:\Malaysia\01_Brueprint\13_Generate_points\02_shift_3ft\merge_all_points_6ftfin_target_shift.shp"
point_poc = r"F:\MAlaysia\Blueprint\POC_Area_20230822\POC_data_202405\PlantingPoint.shp"
terrace_poc = r"F:\MAlaysia\Blueprint\POC_Area_20230822\POC_data_202405\TerraceLining_use.shp"
out_dir = r"D:\Malaysia\01_Brueprint\14_Accuracy\POC"

gdf_point_study = gpd.read_file(point_this_study)
gdf_point_poc = gpd.read_file(point_poc)
use_crs = gdf_point_study.crs

gdf_terrace = gpd.read_file(terrace_poc)
gdf_terrace = gdf_terrace[gdf_terrace.use==1] #select use lines

search_distance = [1, 2, 3, 4, 5]

# dissolve
dissolved_line = unary_union(gdf_terrace.geometry)
gdf_dissolved = gpd.GeoDataFrame([{'geometry': dissolved_line}])
# gdf_dissolved.plot()

result_dic ={}
for sear in search_distance:
    ## create buff with same terrace id
    ## and, number of poc points on this terrace
    buff_terrace = gdf_dissolved.buffer(sear)
    gdf_buff = gpd.GeoDataFrame({'geometry': [buff_terrace.geometry.values[0]]}).set_crs(use_crs)
    
    # count this study
    points_in = gpd.sjoin(gdf_point_study, gdf_buff, how='inner', predicate='within')
    
    result_dic[sear] = [len(points_in)]
            

df = pd.DataFrame(result_dic).T
df["ratio"] = df.loc[:,0] / len(gdf_point_poc)

## to csv
df.to_csv(out_dir + os.sep +"counts_vs_poc.csv")


