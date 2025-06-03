# -*- coding: utf-8 -*-
"""
# Plot palm pixels especially damaged? in ENSO 
"""

import matplotlib.pyplot as plt
import os
import itertools
import rasterio
import rasterio.mask
from rasterio.plot import show
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import pandas as pd
import glob
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
import seaborn as sns


plt.rcParams['font.family'] = 'Times New Roman'

palm_index_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index"
# GOSIF_min_tif = r"F:\MAlaysia\ENSO\01_deviations\lag0\_allseasons\combine_devi_GOSIF_min.tif"
GOSIF_min_tif = r"D:\Malaysia\02_Timeseries\CPA_CPR\4_stressed_pixels\timeeffect\sign_change_morethan1_enso_ratio.tif"
out_dir = os.path.dirname(GOSIF_min_tif) + os.sep + "_png"

poly_dir = r"F:\MAlaysia\AOI\Administration\by_Islands"


""" 
###　まずA1-A4のパームインデックスをmosaic tifのどこに相当するか見つける
## palm indexはグリッドIDに相当なのでグリッドからポイント作成する
"""

if not os.path.isfile(palm_index_dir + os.sep + "palm_index_points_whole.shp"):

    gird_txts = glob.glob(palm_index_dir + os.sep + "*.txt")
    gird_txts = [t for t in gird_txts if not "whole" in t]
    
    points_As = []
    for tx in gird_txts:
        page = os.path.basename(tx)[:-4].split("_")[-1]
        index_list = pd.read_csv(tx, header=None)
        index_list = (index_list[0]).tolist()
    
        gird_shp = palm_index_dir + os.sep + f"grid_01degree_{page}.shp"
        gdf = gpd.read_file(gird_shp) #grid
        gdf_palm = gdf.loc[index_list] # grid with palm
        gdf_points = gdf_palm.geometry.centroid
        
        points_As.append(gdf_points)
    
    
    points_As_ = list(itertools.chain.from_iterable(points_As)) #flatten
    
    gdf_points = gpd.GeoDataFrame({"geometry":points_As_}, crs = "epsg:4326")
    gdf_points.to_file(palm_index_dir + os.sep + "palm_index_points_whole.shp")

else:
    gdf_points = gpd.read_file(palm_index_dir + os.sep + "palm_index_points_whole.shp")
    points_As_ = (gdf_points.geometry.values).tolist()
    
"""
## obtain index of palms of whole image
"""

locations = [(p.xy[0][0],p.xy[1][0]) for p in points_As_]
## target ras
src = rasterio.open(GOSIF_min_tif)
arr = src.read(1)
num_cols = arr.shape[1]

pixel_indices = [src.index(x, y) for x, y in locations]
src.close()

#2Dインデックスを1Dに変換したときの1Dインデックスを拾う
oned_indx_list = [] # this is palm index
for rc in pixel_indices:
    row, col = rc[0], rc[1]
    index_1d = row * num_cols + col
    oned_indx_list.append(index_1d)



# -------------------------------------
"""# Violin plot for palm pixels by regions """
## sensitivity distribution
# -------------------------------------

polys = glob.glob(poly_dir +os.sep +"*.shp")

data_dfs = []
median_dic={}
for poly in polys:
    data_dic ={}
    poly_name = os.path.basename(poly)[:-4]
    gdf = gpd.read_file(poly)
    gdf_diss = gdf.dissolve(by='tmpIs')
    poly_geom = gdf_diss.geometry.values[0]
    
    with rasterio.open(GOSIF_min_tif) as src:
        
        ## obtain index within polygon
        mask = rasterio.features.geometry_mask([mapping(poly_geom)], transform=src.transform, invert=True, out_shape=(src.height, src.width))
        
        # Get the indices of pixels within the polygon
        indices_within_2d = np.argwhere(mask)
        
        #2Dインデックスを1Dに変換したときの1Dインデックスを拾う
        indices_within = [] # this is palm index
        for rc in indices_within_2d:
            row, col = rc[0], rc[1]
            index_1d = row * num_cols + col
            indices_within.append(index_1d)
            
    
    ## combine palm and region index
    indices_use = list(set(oned_indx_list)&set(indices_within))
    
    ## obtain values at indices_within
    arr_ = np.ravel(arr)    
    masked_array = np.array([arr_[i] for i in indices_use])
    masked_array_clean = masked_array[~np.isnan(masked_array)] #no nan
    
    data_dic[poly_name] = masked_array_clean
    
    ## prepare for dataframe
    df = pd.DataFrame(data_dic)
    df_melted = df.melt(var_name=f'{poly_name}', value_name='Values')
    df_melted = df_melted.rename(columns={f"{poly_name}":"region"})
    
    data_dfs.append(df_melted)
    
    ### calcularate median
    median_regi = np.median(masked_array_clean)    
    median_dic[poly_name] = median_regi


df_concat = pd.concat(data_dfs)


fontname='Times New Roman'
plt.rcParams["font.family"] =fontname
f_size_title = 20
f_size = 16

fig =plt.figure(figsize=(10,5))
fig.subplots_adjust(bottom=0.3)
sns.set_style('ticks')
ax = sns.violinplot(x="region", y="Values", data=df_concat, palette="pastel")
plt.tick_params() #labelsize = 30 #軸ラベルの大きさ
# title_str = "Palm SIF decrease by ENSO"
title_str = "ratio: SIF anomaly to relationship change in palm"
plt.title(title_str , fontname=fontname, size=18) #size=50 #グラフのタイトル
plt.ylabel("ratio", fontname = fontname, size=16) #size=50#ｙ軸ラベル
ax.set_ylim(-4, 2)
plt.xlabel("", fontname=fontname)#x軸ラベルの消去
ax.set_xticklabels(ax.get_xticklabels(), fontname=fontname, rotation=10, size=16)

fig.tight_layout()

# outfilename = title_str.replace(" ","_")
outfilename = os.path.basename(GOSIF_min_tif)[:-4]
fig.savefig(out_dir + os.sep + f"{outfilename}_region_palm.png", dpi=600)


### export median to txt
# sort
median_dic_sort = sorted(median_dic.items(), key=lambda x:x[1])

outfile = out_dir + os.sep + f"median_{outfilename}.txt"
with open(outfile, 'w') as f:
    f.writelines(f"{k}\n" for k in median_dic_sort)


