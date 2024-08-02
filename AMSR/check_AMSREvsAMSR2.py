# -*- coding: utf-8 -*-
"""
# AMSRE and AMSR2 may have disperancy
# so plot time series data on palm area growing from 2002
"""

import os
import glob
import rasterio
import geopandas as gpd
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean

datatype = "SM"#"VOD"
glid_shp = r"F:\MAlaysia\AgeEffect\01_test_points\GOSIF_grid_palm_Palmarea2002_only.shp"
AMSRE_dir = rf"E:\Malaysia\AMSRE\01_tif\{datatype}_C\sameday"
AMSR2_dir = rf"D:\Malaysia\AMSR2\DSC\01_tif\{datatype}_C1"
out_dir = r"F:\MAlaysia\ANALYSIS\03_AMSR_comparison"

### first extract palm area (growing since 2002)
gdf_grid= gpd.read_file(glid_shp)
gdf_grid_sel = gdf_grid[gdf_grid["ratio2000"]>0.4] #0.3:187 grids #if >0.4, only 10 grids
# grid to points
gdf_points = gdf_grid_sel.centroid #geoseries
point_list = gdf_points.values


### pick raster val in time series
tifs_amsre = glob.glob(AMSRE_dir + os.sep + "*.tif")
tifs_amsr2 = glob.glob(AMSR2_dir + os.sep + "*.tif")

vales_at_points = []
for i,p in tqdm(enumerate(point_list)):
    xy_list = p.xy
    coords = (xy_list[0][0], xy_list[1][0]) #lon, lat
    
    year_val = {}
    for t in tifs_amsre:
        tifname = os.path.basename(t)[:-4]
        date_str = tifname.split("_")[-1]
        datetime = datetime.strptime(date_str, '%Y%m%d')
        with rasterio.open(t) as src:
            tifval = [x for x in src.sample([coords])][0][0]
            year_val[datetime] = [tifval]
    
    for t in tifs_amsr2:
        tifname = os.path.basename(t)[:-4]
        date_str = tifname.split("_")[3]
        datetime = datetime.strptime(date_str, '%Y%m%d')
        with rasterio.open(t) as src:
            tifval = [x for x in src.sample([coords])][0][0]
            year_val[datetime] = [tifval]
    
    df = pd.DataFrame.from_dict(year_val, columns=[f'p{i}'], orient='index')
    # df = df.drop([2000,2001], axis=0)
    vales_at_points.append(df)       


### convert to monthly data
## merge all point data to one dataframe
df_vales = pd.concat(vales_at_points, axis=1)

df_values_month = df_vales.resample("M").mean()

### Let's plot
# fig = plt.figure()
# ax = fig.add_subplot(111, xlabel="yyyymm", ylabel=datatype, title = "AMSRE to AMSR2")
# pi = df_values_month.loc[:,"p9"]
# x_val = pi.index
# y_val = pi.values
# scatter = ax.plot(x_val, y_val, c="dodgerblue") #scatter
# fig.tight_layout()
# fig.savefig(out_dir + os.sep + f"{datatype}_plot.png")

for i in range(len(vales_at_points)): #plot by column (pi)
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel="yyyymm", ylabel=datatype, title = "AMSRE to AMSR2")
    pi = df_values_month.loc[:,f"p{i}"]
    x_val = pi.index
    y_val = pi.values
    scatter = ax.plot(x_val, y_val, c="dodgerblue") #scatter
    # plt.xlim(200,x_max)
    # plt.ylim(200,y_max)
    # ax.legend(site_list, title='site',loc='upper left',bbox_to_anchor=(1.01, 1.02),fontsize=8) # #
    fig.tight_layout()
    fig.savefig(out_dir + os.sep + f"{datatype}_p{i}_plot.png")


### obtain difference between AMSRE and AMSR2
meds_e = []
meds_2 = []
for i in range(len(vales_at_points)):
    pi = df_values_month.loc[:,f"p{i}"]
    pi_e = pi[((pi.index.year < 2012) & (pi.index.month < 11))]
    pi_2 = pi[(pi.index.year >= 2012) & (pi.index.year < 2023)]
    # calc median
    pi_e_med = pi_e.median()
    pi_2_med = pi_2.median()
    
    if not (pi_e_med ==0) or (pi_2_med==0): # if 0, this dataset is not used
        meds_e.append(pi_e_med)
        meds_2.append(pi_2_med)
    else:
        continue

fin_med_e = mean(meds_e)
fin_med_2 = mean(meds_2)
ratio = fin_med_2 / fin_med_e

df_ratio = pd.DataFrame([fin_med_e, fin_med_2, ratio], index=["AMSRE", "AMSE2","ratio"]).T

outfile = out_dir + os.sep + f"{datatype}_ratio.txt"
df_ratio.to_csv(outfile)



    
    




    
