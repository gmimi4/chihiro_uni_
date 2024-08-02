# -*- coding: utf-8 -*-
"""
# POCデータと比較してみる

"""

import os,sys
import geopandas as gpd
import pandas as pd
from numpy import mean
import numpy as np
import fiona
import shapely
from shapely.geometry import Point, Polygon,MultiPoint,MultiPolygon,LinearRing
from shapely import wkt
from tqdm import tqdm
from statistics import mean
import warnings
warnings.simplefilter('ignore', FutureWarning)

point_this_study = '/Volumes/SSD_2/Malaysia/01_Brueprint/13_Generate_points/02_shift_3ft/merge_all_points_6ftfin_target_shift.shp'
point_poc = '/Volumes/PortableSSD/Malaysia/Blueprint/POC_Area_20230822/POC_data_202405/PlantingPoint.shp'
terrace_poc = '/Volumes/PortableSSD/Malaysia/Blueprint/POC_Area_20230822/POC_data_202405/TerraceLining_use.shp'
# point_this_study = r"D:\Malaysia\01_Brueprint\13_Generate_points\02_shift_3ft\merge_all_points_6ftfin_target_shift.shp"
# point_poc = r"F:\MAlaysia\Blueprint\POC_Area_20230822\POC_data_202405\PlantingPoint.shp"
# terrace_poc = r"F:\MAlaysia\Blueprint\POC_Area_20230822\POC_data_202405\TerraceLining_use.shp"
out_dir = '/Volumes/SSD_2/Malaysia/01_Brueprint/14_Accuracy/POC'

gdf_point_study = gpd.read_file(point_this_study)
gdf_point_poc = gpd.read_file(point_poc)
use_crs = gdf_point_study.crs

# gdf_terrace = gpd.read_file(terrace_poc)
# gdf_terrace = gdf_terrace[gdf_terrace.use==1] #select use lines


""" #connect disconnected lines to one line """
def filter_feature(feature):
    # Example criteria: select features where the property 'id' is 1
    return feature['properties'].get('use') == 1

# Open the shapefile and filter features
# src = fiona.open(terrace_poc)
with fiona.open(terrace_poc) as src:
    crs = src.crs
    driver = src.driver
    filtered_features = [feature for feature in src if filter_feature(feature)]


merged_geometry = shapely.ops.linemerge([shapely.geometry.shape(feature["geometry"]) for feature in filtered_features])
# merged_geometry = shapely.ops.linemerge([shapely.geometry.shape(feature["geometry"]) for feature in src])

schema = {
        'geometry': 'LineString',  # Assuming the merged geometry is of type LineString
        'properties': {}           # No properties in this case
    }

outmerge_dir = os.path.dirname(terrace_poc)
outmergefile = outmerge_dir + os.sep + os.path.basename(terrace_poc)[:-4] + "_merge.shp"
# with fiona.open(outmergefile, 'w', driver=driver, crs=crs, schema=schema) as dst:
#     dst.write({
#             'geometry': shapely.geometry.mapping(merged_geometry),
#             'properties': {}
        # })


## read merged lines
gdf_terrace = gpd.GeoDataFrame({"geometry":[merged_geometry]}, crs = crs)
## break Multi to single
gdf_terrace = gdf_terrace.explode()
gdf_terrace.to_file(outmergefile)


gdf_terrace = gdf_terrace.reset_index().drop("level_0", axis=1)
gdf_terrace = gdf_terrace.reset_index().drop("level_1", axis=1)


## create buffer for merged line
search_distance = 3
## create buff with same terrace id
buff_dic = {}
for i,row in gdf_terrace.iterrows():
    buff = row.geometry.buffer(search_distance)
    buff_dic[i] = buff

# search_buff = gdf_terrace.buffer(search_distance)
gdf_buff = gpd.GeoDataFrame(index = buff_dic.keys(), geometry=list(buff_dic.values())).set_crs(use_crs)

result_dic ={} # terraceID: [mean distance, count study, count poc]
# for i,row in gdf_buff.iterrows():
for i,row in gdf_terrace.iterrows(): #pic poc points by 1m buffer
    
    buff_1m = row.geometry.buffer(1)
    buff_poc = row.geometry.buffer(search_distance)
    
    gdf_buff_1m = gpd.GeoDataFrame({"geometry":[buff_1m]}).set_crs(use_crs)
    gdf_buff_poc = gpd.GeoDataFrame({"geometry":[buff_poc]}).set_crs(use_crs)
    
    """ # extract points by poc terrace buff"""
    gdf_point_study_use = gpd.sjoin(gdf_point_study, gdf_buff_poc, how='inner', predicate='within') #capture by 5m buff
    gdf_point_poc_use = gpd.sjoin(gdf_point_poc, gdf_buff_1m, how='inner', predicate='within') # caoture by 1m buff
        
    """ # find nearest point from poc and its distance"""
    # gdf_point_study_use["Processed"]=0
    
    nearest_distance_list=[]
    for ip,rowp in tqdm(gdf_point_poc_use.iterrows()):       
            poi_poc = rowp.geometry
            distances = gdf_point_study_use.distance(poi_poc) #pocからstudy pointsまでの距離#series
            
            if len(distances)>0:
                distance_nearest = min(distances) #最近傍距離
            else:
                distance_nearest=np.nan
            
            # if nearest_idx not in used_study_idx: #まだ抽出されていないstudy pointであれば
            #     used_study_idx.append(nearest_idx)
            #     nearest_distance_list.append(distance_nearest)
            # else:

            nearest_distance_list.append(distance_nearest)
    
    mean_distance = np.nanmean(nearest_distance_list)
                
    count_study = len(gdf_point_study_use)
    count_poc = len(gdf_point_poc_use)
    
    result_dic[i] = [mean_distance, count_study, count_poc]
    

""" #export to csv """
### Tableを作成
df = pd.DataFrame(result_dic.values(), columns=["meandistance","count_study","count_poc"])

#export csv
outfile = out_dir + os.sep + "acuracy_poc_comparison_by_line.csv"
df.to_csv(outfile)


""" # join lineshpe and export"""
gdf_terrace_join = gdf_terrace.merge(df, left_index=True, right_index=True)
gdf_terrace_join.to_file(out_dir + os.sep + os.path.basename(terrace_poc)[:-4] + "_merge_accuracy.shp")
