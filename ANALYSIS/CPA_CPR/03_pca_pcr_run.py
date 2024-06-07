# -*- coding: utf-8 -*-
"""
#https://www.kaggle.com/code/gianinamariapetrascu/pca-varimax-rotation
# https://www.youtube.com/watch?v=BiuwDI_BbWw
別のpca_pcr pyに渡してrelative importanceの辞書を受け取る

#Affine: https://www.perrygeo.com/python-affine-transforms.html
"""
# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install factor_analyzer

import numpy as np
import pandas as pd
import os,sys
import glob
import geopandas as gpd
from tqdm import tqdm
import rasterio
from rasterio.crs import CRS
with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
    rio_crs = CRS.from_epsg(4326)
    proj_crs = CRS.from_user_input(rio_crs)

# import subprocess
os.chdir(r"C:\Users\chihiro\Desktop\Python\ANALYSIS\CPA_CPR")
import pca_pcr

in_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\1_vars_at_pixels\A1"
PageName = os.path.basename(in_dir)
csv_file_list = glob.glob(in_dir + "\\*.csv")
p_val = 0.1
out_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras" + os.sep + "p_" + str(p_val).replace(".", "")
os.makedirs(out_dir, exist_ok=True)

startyear = 2002
endyear = 2010

# ## extentをつくったポリゴンでtransformを取得する
# Malaysia_land_shape = r"C:\Users\chihiro\Desktop\PhD\Malaysia\AOI\Administration\National_boundary\Malaysia_national_boundary.shp"
# gdf_land_Malaysia = gpd.read_file(Malaysia_land_shape)
# buff_land = gdf_land_Malaysia.buffer(0.3)
# bounds = buff_land.bounds
# xmin, ymin, xmax, ymax = bounds["minx"][0],bounds["miny"][0],bounds["maxx"][0],bounds["maxy"][0]
# polygon_geom = Polygon([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
# extent_polygon = gpd.GeoDataFrame(geometry=[polygon_geom], crs=proj_crs)

### time series csvをつくったラスターのどれか
# sample_tif = r"D:\Malaysia\GPM\01_tif\extent\MERG_20040904_extent.tif" #→Affineが変
sample_tif = r"F:\MAlaysia\SIF\GOSIF\02_tif_age_adjusted\res_01\extent\GOSIF_2000081_extent_adj_res01_extent.tif"
with rasterio.open(sample_tif) as src:
    src_arr = src.read(1)
    meta = src.meta
    transform = src.transform
    height, width = src_arr.shape[0], src_arr.shape[1]
    # profile = src.profile
    
## 参照しているラスターのAffineのheight pixelはマイナスになっていてほしい
meta.update({"nodata":np.nan})


use_vars = ["rain","temp","VPD","Et","Eb","SM","VOD", "p_values"]

"""### relative importanceをピクセルごとに集計する """
idx_importance_dic = {}
for csv_file in tqdm(csv_file_list):
    
    idx = os.path.basename(csv_file)[:-4]
    # csv_file = r"D:\Malaysia\02_Timeseries\CPA_CPR\1_vars_at_pixels\A1\13310.csv"
    try:
        relative_importances = pca_pcr.main(csv_file, p_val, startyear, endyear)
    except:
        relative_importances = dict()
    finally: #なぜか空dictに更新されなかったのでここで実行
        idx_importance_dic[idx] = relative_importances
    # idx_importance_dic[idx] = relative_importances

"""### 変数ごとにラスターに変換 """
for vari in use_vars:
    # vari = "rain"
    ras_dic = {}
    for i,imoprtance_dic in idx_importance_dic.items():
        # i=4446
        # imoprtance_dic = idx_importance_dic[str(i)]
        indx = int(i)
        if len(imoprtance_dic)>0:
            var_importance = imoprtance_dic[vari]
            ras_dic[indx] = var_importance
        else:
            ras_dic[indx] = np.nan
    #test_vals = ras_dic.values()
    
    #念のためindx順にソート
    ras_dic_sort = sorted(ras_dic.items()) #タプルになった(indx, importance)
    # importance_arr = np.array([r[1] for r in ras_dic_sort])
    
    #これに入れる
    importance_arr = np.full(len(ras_dic_sort), np.nan)
    
    #test
    # test_array = np.linspace(0, 8, 9)
    # test_re = test_array.reshape((3, 3))
    # test_flatten = np.ravel(test_re)
    
    for i in ras_dic_sort:
        arri = i[0]
        arrval = i[1]
        np.put(importance_arr, [arri], arrval)
        
    
    # reshape
    importance_arr_re = importance_arr.reshape((height, width))
    
    out_file = out_dir +f"\\{vari}_importance_{str(startyear)}-{str(endyear)}.tif"
    with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
        with rasterio.open(out_file, 'w', **meta) as dst:
          dst.write(importance_arr_re, 1)


