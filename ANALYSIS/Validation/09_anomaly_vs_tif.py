# -*- coding: utf-8 -*-
"""
Compare FFB anomalies due to ENSO and tif values such as recovery
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
import math

os.chdir(r"C:\Users\chihiro\Desktop\Python\ANALYSIS\YieldWater")
import _yield_csv 


shp_region = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
shp_extent = r"F:\MAlaysia\AOI\extent\Malaysia_and_Indonesia_extent_divided.shp"
shp_01grid_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index"
shp_grid = shp_01grid_dir + os.sep + "grid_01degree_210_496.shp"
palm_txt2002 = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index\grid_01degree_210_496_palm2002.txt"
csv_anomaly = r"D:\Malaysia\02_Timeseries\YieldWater\04_yield_with_ENSO\_detr\ENSOIOD_yield_detr_anomaly.csv"
out_dir = r"D:\Malaysia\Validation\8_anomaly_correlation\_detrended"

# tifpath = r"D:\Malaysia\02_Timeseries\Resilience\07_perturbation\_mosaic\mosaic_recoveryrate.tif"
tifpath = r"D:\Malaysia\02_Timeseries\Resilience\01_ARX\EVI16days\_seasonl\lag_1\_mosaic\mosaic_EVIt-1_importanceARX_2002-2023.tif"
# tifpath = r"D:\Malaysia\02_Timeseries\Resilience\01_ARX\_seasonl\lag_1\_mosaic\mosaic_GOSIFt-1_importanceARX_2002-2023.tif"
# tifpath = r"D:\Malaysia\02_Timeseries\Resilience\04_recovery\timerange3\mean_time\_mosaic\mosaic_recovery_meantime.tif"
valname = "Autot1_EVI16_season"
# valname = "Autot1_season"
# valname = "Recovery3"

""" #Yield"""
df_yield, df_yield_z, df_yield_detr = _yield_csv.main()
nondata_region = list(df_yield_z[df_yield_z.isna().all(axis=1)].index)
nondata_region = nondata_region +["Maluku Utara"] #これがいる

""" #Anomlay"""
df_anomaly = pd.read_csv(csv_anomaly, index_col=0)
df_anomaly = df_anomaly.dropna()


""" set palm """
df_palm = pd.read_csv(palm_txt2002, header=None)
list_palm = df_palm[0].values.tolist()


""" prepare polygon"""
## regions
gdf_region = gpd.read_file(shp_region)
gdf_region = gdf_region.drop(columns=['NAME_1','ADM1_EN','rank_yield']) #'slope',
gdf_region = gdf_region.set_index("Name")
## grid
gdf_grid = gpd.read_file(shp_grid)


""" # pick tif values at palm pixels since 2002"""
ras_values_result = []
for regi, row in tqdm(gdf_region.iterrows()):
    # regi_poly = gdf_region.loc[0,:].geometry
    # print(regi)
    if regi in nondata_region:
        continue
    else:
        # print(regi)
        regi_poly = row.geometry
        gdf_regi = gpd.GeoDataFrame({"geometry":[regi_poly]}).set_crs(gdf_region.crs) #multipolygon
        
        """ # select target grids"""
        ## grids which intersect with region polygon
        gdf_tar_grid = gdf_grid[gdf_grid.intersects(regi_poly)] #index is target tif filename
        ### select grid id within palm 2
        gdf_tar_grid = gdf_tar_grid[gdf_tar_grid['raster_val'].isin(list_palm)] #palm 2002
        
        """ # convert to points"""
        gdf_points =  gpd.GeoDataFrame(gdf_tar_grid.copy(), geometry=gdf_tar_grid.geometry.centroid)
        ## test
        # gdf_points.to_file(out_dir +os.sep +f"{regi}_points.shp")
        
        """ # collect pixel values overlaying with points
            # Since rater size differs from grid size,,, points are used to pick values one by one"""
        ras_values = {}
        src = rasterio.open(tifpath)
        ras_values_year =[]
        for point in gdf_points.geometry:
            row, col = src.index(point.x, point.y)
            raster_value = src.read(1)[row, col]
            if not math.isnan(raster_value): 
                ras_values_year.append(raster_value)
        
        ras_values[regi] = ras_values_year
        src.close()
        
        """ # mean SIF values"""
        try:
            ras_values_mean = {key: sum(values) /len(values) for key, values in ras_values.items()}
        except:
            ras_values_mean = {key: np.nan for key, values in ras_values.items()}
        ## convert to dataframe
        df_rasval = pd.DataFrame(ras_values_mean, index=[0])
        df_rasval = df_rasval.rename(index={0:regi},columns={regi:f"{valname}"})
        ras_values_result.append(df_rasval)



""" export SIF sum result"""
df_sif_concat = pd.concat(ras_values_result)
df_sif_concat.to_csv(out_dir + os.sep + f"{valname}.csv")

## ------------------------------------------
# Simple correlation of values in each year
## ------------------------------------------

""" prepare dataset"""
tarvar = "anomalyENSO"
df_anomaly_use = df_anomaly[[tarvar]]

df_dataset = pd.concat([df_anomaly_use, df_sif_concat], axis=1)
df_dataset = df_dataset.dropna()

### Calculate Pearson correlation between two columns
corr_simple, p_value_simpe = pearsonr(df_dataset[tarvar], df_dataset[valname])

## Plot
plt.figure(figsize=(5, 5))
plt.scatter(df_dataset[valname], df_dataset[tarvar], label=f'{tarvar} vs {valname} time')
slope, intercept = np.polyfit(df_dataset[valname], df_dataset[tarvar], 1)
regression_line = slope * df_dataset[valname]+ intercept
plt.plot(df_dataset[valname], regression_line, color='grey',linestyle='--', label='Least Squares Fit')
plt.xlabel(f'{valname}', fontsize=16)
plt.ylabel('FFB anomaly', fontsize=16)
plt.legend()
plt.text(0.05,0.2,'$ r $=' + str(round(corr_simple, 4)),fontsize=14, transform=plt.gca().transAxes)
plt.text(0.05,0.1,'$ p $=' + str(round(p_value_simpe, 4)),fontsize=14, transform=plt.gca().transAxes)
plt.tight_layout()
plt.show()
plt.savefig(out_dir + os.sep + f"{tarvar}_vs_{valname}.png")


## ------------------------------------------
# Spearman correlation 
## ------------------------------------------
## Calc ranks
df_dataset[tarvar] = df_dataset[tarvar].rank(ascending=False)
df_dataset[f"rank_{valname}"] = df_dataset[valname].rank(ascending=False)
# Calculate Spearman correlation
corr_rank, p_value_rank = spearmanr(df_dataset[tarvar], df_dataset[f"rank_{valname}"] )    

outxt = out_dir + os.sep + f"spearman_{valname}.txt"
with open(outxt,"w") as f:
    f.write(f"rank_corr: {corr_rank}, rank_pval: {p_value_rank}")


