# -*- coding: utf-8 -*-
"""
Box plot
Extract palm area by states or provinces
# Cartopy is in gdal_copy (spyder ok) and rasterio_copy2 (no spyder)
"""
# -------------------------------------
"""# Box plot by regions """
# -------------------------------------

import matplotlib.pyplot as plt
import os
import itertools
import rasterio
import rasterio.mask
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import pandas as pd
import glob
import seaborn as sns
plt.rcParams['font.family'] = 'Times New Roman'

tif_path = r"D:\Malaysia\02_Timeseries\Resilience\03_halfperiod\_mosaic\mosaic_psdall_half.tif"
out_dir = os.path.dirname(tif_path) + os.sep + "_png"

palm_index_txt = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index\palm_index_shape_210_496.txt"
txt_dir_Malaysia = r"F:\MAlaysia\AOI\Administration\Malaysia\Index"
txt_dir_Indone = r"F:\MAlaysia\AOI\Administration\Indonesia\Index"

regions_txts_malay = glob.glob(txt_dir_Malaysia + os.sep + "*.txt")
regions_txts_indone = glob.glob(txt_dir_Indone + os.sep + "*.txt")

### make list of palm index
df_palm_index = pd.read_csv(palm_index_txt, header=None)
index_list_palm = df_palm_index.iloc[:,0].tolist()

### Make raster to 1d array
with rasterio.open(tif_path) as src:
    arr = src.read(1)
    arr_1d = np.ravel(arr)


# values_regi = {}
# regions = []
# arr_vals = []
def region_process(regions_txts):
    for txt in regions_txts:
        try:
            ### if txt contains data        
            df_region = pd.read_csv(txt, header=None)
        except:
            continue
        index_list_regi = df_region.iloc[:,0].tolist()
        regi_name = os.path.basename(txt)[:-4].replace("_"," ")
        
        """ # combine palm and region index """
        index_list_both = list(set(index_list_palm)&set(index_list_regi))
        
        if len(index_list_both) ==0:
            continue
        else:
            """ collect tif values at common index """
            arr_regi = arr_1d[index_list_both]
            arr_regi = arr_regi[~np.isnan(arr_regi)] #no nan
            values_regi[regi_name] = arr_regi
            
            regions.append(regi_name)
            arr_vals.append(arr_regi)
            
    return regions, arr_vals, values_regi


""" # Process. add list Malaysia then Indonesia"""
## change order
regions_txts_malay = sorted(regions_txts_malay, key=lambda x: os.path.basename(x), reverse=True,)
regions_txts_indone = sorted(regions_txts_indone, key=lambda x: os.path.basename(x), reverse=True,)

values_regi = {}
regions = []
arr_vals = []
## Malaysia
regions, arr_vals, values_regi = region_process(regions_txts_malay)
## Indonesia
regions, arr_vals, values_regi = region_process(regions_txts_indone)


"""# Box plot by regions """

arr_1d_nonan = arr_1d[~np.isnan(arr_1d)]
minval = np.percentile(arr_1d_nonan, 1)
maxval = np.percentile(arr_1d_nonan, 99)
meanval = arr_1d_nonan.mean()
minval = minval - abs(meanval*0.75) #for visualization
maxval = maxval + abs(meanval*0.75)

### name of label
tifname = os.path.basename(tif_path)[:-4]
if "Diff" in tifname:
    if "psd" in tifname:
        labelname = "change in beta"
    if "MKDiff" in tifname:
        labelname = "change in slope"
    if "mean" in tifname:
        labelname = "change in mean"
    if "std" in tifname:
        labelname = "change in std"
    if "MKpDiff" in tifname:
        labelname = "change in slope from beta "
else:
    if "psd" in tifname:
        labelname = "beta"
    if "MKall" in tifname:
        labelname = "slope"
    if "mean" in tifname:
        labelname = "mean"
    if "std" in tifname:
        labelname = "std"
    if "MKpall" in tifname:
        labelname = "slope from beta"
    

fontname='Times New Roman'
plt.rcParams["font.family"] =fontname
plt.tick_params() #labelsize = 30 #軸ラベルの大きさ
f_size_title = 20
f_size = 16

fig =plt.figure(figsize=(10,50), tight_layout=True) #
ax = fig.add_subplot(1, 1, 1)
ax.boxplot(arr_vals, labels=regions, vert=False) #hrizontal
ax.set_xlabel(labelname, size=16)
ax.set_xticklabels(ax.get_xticklabels(), fontname=fontname, size=14)
# ax.set_xlim(minval, maxval)
# ax.set_ylabel('values')
# ax.set_ylim(minval, maxval)
ax.set_yticklabels(ax.get_yticklabels(), fontname=fontname, size=12)
# title_str = tifname.replace("_"," ")
# plt.title(title_str , fontname=fontname, size=18) #size=50 #グラフのタイトル
plt.show()

## Save fig
fig.savefig(out_dir + os.sep + f"{tifname}_region.png", dpi=600)

""" # Export median text"""
median_dic = {}
for regi, vals in values_regi.items():
    medianval = np.nanmedian(vals)
    median_dic[regi] = medianval

sorted_median = sorted(median_dic.items(), key=lambda x:x[1])

#txtで出力
out_txt = os.path.join(out_dir,f"{tifname}_region.txt")
with open(out_txt, mode='w') as f:
    f.writelines(f"{k}\n" for k in sorted_median)
