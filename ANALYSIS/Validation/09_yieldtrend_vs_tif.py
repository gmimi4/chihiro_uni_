# -*- coding: utf-8 -*-
"""
Compare FFB trend and tif values such as recovery
read csv after anomaly comparison
"""
import os
import glob
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

os.chdir(r"C:\Users\chihiro\Desktop\Python\ANALYSIS\YieldWater")
import _yield_csv 


# shp_region = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
# shp_extent = r"F:\MAlaysia\AOI\extent\Malaysia_and_Indonesia_extent_divided.shp"
# shp_01grid_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index"
# shp_grid = shp_01grid_dir + os.sep + "grid_01degree_210_496.shp"
# palm_txt2002 = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index\grid_01degree_210_496_palm2002.txt"
# # csv_anomaly = r"D:\Malaysia\02_Timeseries\YieldWater\04_yield_with_ENSO\ENSOIOD_yield_anomaly.csv"
out_dir = r"D:\Malaysia\Validation\8_slope_correlation\_detrended"
csv_dir = r"D:\Malaysia\Validation\8_anomaly_correlation\_detrended"
valname = "sd_EVI16"
csv_recovery = csv_dir + os.sep + f"{valname}.csv"


""" #Yield"""
df_yield, df_yield_z, df_yield_detr = _yield_csv.main()
nondata_region = list(df_yield_z[df_yield_z.isna().all(axis=1)].index)
nondata_region = nondata_region +["Maluku Utara"] #これがいる

""" #Read recovery values"""
df_recovery = pd.read_csv(csv_recovery, index_col=0)
df_sif_concat = df_recovery

## ------------------------------------------
# Slope correlation 
## ------------------------------------------
slope_result = {}
df_yield_valid = df_yield.dropna(how="all")
for regi, yilds in df_yield_valid.iterrows():
    ## first extract ras val
    df_sif_regi = df_sif_concat.loc[regi,:]
    rasval = df_sif_regi.values[0]
    if not abs(rasval) >=0:
        continue
    else:
        ### set index int
        yilds.index = yilds.index.astype(int)
        ### sort index
        yilds = yilds.sort_index()
        yilds = yilds.dropna()
        ### calc slope
        slp_yild, intercept = np.polyfit(np.arange(len(yilds)), yilds, 1)
        slope_result[regi] = [slp_yild, rasval]
    

df_slope_result = pd.DataFrame.from_dict(slope_result).T
df_slope_result = df_slope_result.rename(columns={0:"yield",1:valname})

### Calculate Pearson correlation between two columns
corr_slp, p_value = pearsonr(df_slope_result['yield'], df_slope_result[valname])

## Plot
plt.figure(figsize=(5, 5))
plt.scatter(df_slope_result[valname], df_slope_result['yield'], label=f'annaul slope vs {valname}')
# Calculate the least squares fit (linear regression)
slope, intercept = np.polyfit(df_slope_result[valname], df_slope_result['yield'], 1)
regression_line = slope * df_slope_result[valname]+ intercept
plt.plot(df_slope_result[valname], regression_line, color='grey',linestyle='--', label='Least Squares Fit')
plt.xlabel(valname, fontsize=16)
plt.ylabel('FFB slope', fontsize=16)
plt.legend()
plt.text(0.1,0.8,'$ r $=' + str(round(corr_slp, 4)),fontsize=14, transform=plt.gca().transAxes)
plt.text(0.1,0.7,'$ p $=' + str(round(p_value, 4)),fontsize=14, transform=plt.gca().transAxes)
plt.tight_layout()
plt.show()
plt.savefig(out_dir + os.sep + f"slope_vs_{valname}.png")


## ------------------------------------------
# Slope Spearman correlation 
## ------------------------------------------
## Calc ranks
df_slope_result["rank_yield"] = df_slope_result["yield"].rank(ascending=False)
df_slope_result[f"rank_{valname}"] = df_slope_result[valname].rank(ascending=False)
# Calculate Spearman correlation
corr_rank, p_value_rank = spearmanr(df_slope_result["rank_yield"], df_slope_result[f"rank_{valname}"])   

outxt = out_dir + os.sep + f"spearman_{valname}.txt"
with open(outxt,"w") as f:
    f.write(f"rank_corr: {corr_rank}, rank_pval: {p_value_rank}")