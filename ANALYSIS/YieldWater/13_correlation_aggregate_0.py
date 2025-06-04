# -*- coding: utf-8 -*-
"""
Sum pearson correlation for each var to find overall contribution
First var, second ENSO
"""
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import seaborn as sns


pp = "_pearson_detr_0"
# pp="_pearson_0"
pearson_dir = rf"D:\Malaysia\02_Timeseries\YieldWater\01_correlation_timelag\{pp}" #var
out_dir = rf"D:\Malaysia\02_Timeseries\YieldWater\09_Strategy\01_overall_correlation\{pp}"
os.makedirs(out_dir, exist_ok=True)
pp2 = pp
pearson_dir_ENSO = rf"D:\Malaysia\02_Timeseries\YieldWater\06_correlation_timelag_ENSO\{pp2}"
# out_dir_ENSO = rf"D:\Malaysia\02_Timeseries\YieldWater\09_Strategy\01_overall_correlation\{pp2}_ensoiod"
# os.makedirs(out_dir_ENSO, exist_ok=True)
shp_region = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"

varlist = ['rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD'] #'GOSIF', 

pearson_csvs = glob.glob(pearson_dir + os.sep + "*_abs.csv")

""" vars """
overall_regions = []
for csv_pearson in tqdm(pearson_csvs):
    # i=37
    # row = gdf_region.loc[i,:]
    reginame_csv = os.path.basename(csv_pearson)[:-4].split("_")[1]

    df_peason = pd.read_csv(csv_pearson, index_col=0)
    ## drop GOSIF
    # df_peason = df_peason.drop("GOSIF", axis=1)
    
    overall = {}
    for var in varlist:
        seri_var = df_peason.loc[:,var]
        # seri_var_sum = np.nansum(seri_var.values)
        seri_var_mean = np.nanmean(seri_var.values)
        overall[var] = [seri_var_mean]

    df_overall = pd.DataFrame.from_dict(overall)    
    df_overall.index = [reginame_csv]
    overall_regions.append(df_overall)

    
df_overall_regions = pd.concat(overall_regions)

## sort
# csv_sample = rf"D:\Malaysia\02_Timeseries\YieldWater\03_var_variation\_pearson_neg\std_slope.csv"
csv_sample = rf"D:\Malaysia\02_Timeseries\YieldWater\sample_order.csv"
df_sample = pd.read_csv(csv_sample, index_col=0, header=None)
df_sample.index = df_sample.index.str.replace(" ", "", regex=False)

df_overall_regions = df_overall_regions.reindex(df_sample.index)
## Export 
df_overall_regions.to_csv(out_dir + os.sep + "overall_correlation.csv")


# """ #Plot time series all vars -> no need because of use of Excel"""

# for i, row in df_overall_regions.iterrows():#df_overall_regions_copy.iterrows():
#     row_data = df_overall_regions.loc[[i]]
#     fig,axes = plt.subplots(figsize=(8, 2))
#     fig.subplots_adjust(hspace=0.5) 
#     vmin = row_data.min(axis=1).values[0]
#     vmax = row_data.max(axis=1).values[0]
    
#     ax = axes#[i]
#     # if i >= 0:
#     sns.heatmap(row_data, annot=True, cmap="coolwarm", cbar=True, vmin=vmin, vmax=vmax, center=0)
#     ax.xaxis.set_label_position('top')
#     ax.xaxis.tick_top()
    
#     fig.savefig(out_dir + os.sep + f"{i}_overallcorr")
#     plt.close()

""" #ENSO/IOD"""
pearson_csvs_enso = glob.glob(pearson_dir_ENSO + os.sep + "*_abs.csv")

overall_regions = []
for csv_pearson in tqdm(pearson_csvs_enso):
    # i=37
    # row = gdf_region.loc[i,:]
    reginame_csv = os.path.basename(csv_pearson)[:-4].split("_")[1]

    df_peason = pd.read_csv(csv_pearson, index_col=0)
    
    enso_iod = df_peason.columns.tolist() #obtain strings
    overall = {}
    for var in enso_iod: #['ENSO', 'IOD']
        seri_var = df_peason.loc[:,var]
        # seri_var_sum = np.nansum(seri_var.values)
        seri_var_mean = np.nanmean(seri_var.values)
        overall[var] = [seri_var_mean]
    
    df_overall = pd.DataFrame.from_dict(overall)    
    df_overall.index = [reginame_csv]
    overall_regions.append(df_overall)

    
df_overall_regions = pd.concat(overall_regions)

## sort
df_overall_regions = df_overall_regions.reindex(df_sample.index)
## Export 
df_overall_regions.to_csv(out_dir + os.sep + "overall_correlation_ensoiod.csv")
        
# """ #Attach to poly"""
gdf_region = gpd.read_file(shp_region)
gdf_region = gdf_region.drop(["NAME_1","slope","ADM1_EN","rank_yield"],axis=1)
gdf_region["Name_"] = gdf_region["Name"].str.replace(" ","")

df_overall_regions = df_overall_regions.reset_index()
df_overall_regions = df_overall_regions.rename(columns={"index":"Name_"})
gdf_region_merge = pd.merge(gdf_region, df_overall_regions, on="Name_", how='inner')
gdf_region_merge.to_file(out_dir + os.sep + "overall_correlation_ensoiod.shp")

        


