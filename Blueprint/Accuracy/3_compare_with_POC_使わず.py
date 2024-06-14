# -*- coding: utf-8 -*-
"""
# POCデータと比較してみる

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
warnings.simplefilter('ignore', FutureWarning)

point_this_study = r"D:\Malaysia\01_Brueprint\13_Generate_points\02_shift_3ft\merge_all_points_6ftfin_target_shift.shp"
point_poc = r"F:\MAlaysia\Blueprint\POC_Area_20230822\POC_data_202405\PlantingPoint.shp"
terrace_poc = r"F:\MAlaysia\Blueprint\POC_Area_20230822\POC_data_202405\TerraceLining_use.shp"

gdf_point_study = gpd.read_file(point_this_study)
gdf_point_poc = gpd.read_file(point_poc)
use_crs = gdf_point_study.crs

gdf_terrace = gpd.read_file(terrace_poc)
gdf_terrace = gdf_terrace[gdf_terrace.use==1] #select use lines

search_distance = 5
## create buff with same terrace id
buff_dic = {}
for i,row in gdf_terrace.iterrows():
    buff = row.geometry.buffer(search_distance)
    buff_dic[i] = buff

# search_buff = gdf_terrace.buffer(search_distance)
gdf_buff = gpd.GeoDataFrame(index = buff_dic.keys(), geometry=list(buff_dic.values())).set_crs(use_crs)

result_dic ={} # mean distance
for i,row in gdf_buff.iterrows():
    
    buff_poc = gpd.GeoDataFrame({"geometry":[row.geometry]}).set_crs(use_crs)
    
    """ # extract generated points by poc terrace for comparison"""
    gdf_point_study_use = gpd.sjoin(gdf_point_study, buff_poc, how='inner', predicate='within')
    gdf_point_poc_use = gpd.sjoin(gdf_point_poc, buff_poc, how='inner', predicate='within')
        
    """ # find nearest point from poc and its distance"""
    # gdf_point_study_use["Processed"]=0
    
    nearest_distance_list=[]
    used_study_idx =[]
    for ip,rowp in tqdm(gdf_point_poc_use.iterrows()):       
            poi_poc = rowp.geometry
            distances = gdf_point_study_use.distance(poi_poc) #pocからstudy pointsまでの距離#series 
            distance_nearest = min(distances) #最近傍距離
            nearest_idx = distances.idxmin() #そのstudy pointのidx
            
            if nearest_idx not in used_study_idx: #まだ抽出されていないstudy pointであれば
                used_study_idx.append(nearest_idx)
                nearest_distance_list.append(distance_nearest)
            else:
                
            
            ## processed=1
            gdf_point_study_use.at[nearest_idx,"Processed"]=1
            
            #距離が短い順に並べる
            sorted_dict = dict(sorted(dis_dic.items(), key=lambda item: item[1]))
            min_distance = min(list(dis_dic.values()))
            select_p = [p for p, d in dis_dic.items() if d == min_distance][0]
            
            count_list.append([select_p, min_distance]) #pになってたのを修正

            """
            for p,d in sorted_dict.items():
                check_process = points_within[points_within.geometry==p].Processed.values[0]
                count_list.append(p) #checkなし
                # if check_process ==0:
                #     count_list.append(p)
                #     break
                # else:
                #     continue #最短のポイントのProcessedが1だったら次に近いポイントへ
            """
        else:
            # print("sparse")
            min_distance=99
            # continue
        
        ref_dic[iv] = min_distance #geometry pointを入れると謎に数が足りなくなる、rowはSeriesで入れられない       
        # points_within_valid = points_within[points_within.Processed==0]
        # points_within_list = points_within_valid.geometry.tolist()
        # count_list.append(len(points_within_list)) #validation point1つに対して1つのgenerated pointがひっかかるか
    
    # count_dic[close_limit] = sum(count_list)
    # count_dic[close_limit] = len(count_list)
    count_dic[close_limit] = count_list
    

## cehck
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 5))
    
# for li in [line_t2]:
#   linestring_x, linestring_y = li.xy
#   ax.plot(linestring_x, linestring_y, color='black', label='Linestring')
    
for p in [buff]: #, neardict[1][0], neardict[2][0]
    x,y = p.exterior.xy
    ax.plot(x, y,color='green', label='Polygon')
    

gdf_point_study_use_list = [p.geometry for i,p in gdf_point_study.iterrows()]
for p in gdf_point_study_use_list: #midpointはどれも同じ
  x,y = p.xy
  ax.plot(x[0], y[0], 'bo',  label='Point')
  
# for p in points_lineT2: #midpointはどれも同じ
#   x,y = p.xy
#   ax.plot(x[0], y[0], 'bo',  label='Point')



### Tableを作成
# df_table = pd.DataFrame(count_dic.values(), index=count_dic.keys()).T
numlist = [len(n) for n in count_dic.values()]
df_table = pd.DataFrame(numlist, index=count_dic.keys())

### total num of validation points
total_num = len(gdf_validation)
# df_table.loc["ratio"] = df_table.iloc[0]/total_num
df_table.loc[:,"ratio"] = df_table.iloc[:,0]/total_num
df_table = df_table.T

### mean  (あとで変えて)
# d_list = [d[1] for d in count_list]
d_list = [d for k,d in ref_dic.items()]
mean_dis = mean(d_list)

### point  (あとで変えて) ##これdistanceバッファーで複数ひっかかったポイントは重複あり
# p_list = [d[0].geometry for d in count_list]
# gdf_result_point = gpd.GeoDataFrame({"geometry":p_list,"distance":d_list},crs="epsg:32648")

## referenceのポイントに結合する場合
df_result_distance = pd.DataFrame({"distance":d_list}, index=ref_dic.keys())
gdf_result_point = pd.concat([gdf_validation, df_result_distance], axis=1)
gdf_result_point = gdf_result_point.drop(["buffer"], axis=1)

###出力をreference pointにした方が分かりやすいかも


#export csv
outfile = out_dir + "\\accuracy_unique.csv"
df_table.to_csv(outfile)

outfile_shp = out_dir + "\\accuracy_unique.shp" #generated pointがreference pointの数だけ出る
gdf_result_point.to_file(outfile_shp)
