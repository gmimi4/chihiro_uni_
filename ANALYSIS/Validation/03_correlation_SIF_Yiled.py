# -*- coding: utf-8 -*-
"""
Compare Yield data and SIF annual sum 0.1 degree by provinces or states
"""
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
# from rasterstats import zonal_stats
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import scipy.stats
import matplotlib.pyplot as plt
import glob
import numpy as np
import statistics
import math

yield_csv_malay = r"D:\Malaysia\Validation\1_Yield_doc\Malaysia\Malaysia.csv"
yield_csv_indone = r"D:\Malaysia\Validation\1_Yield_doc\Indonesia\Indonesia_CPO.csv"
shp_malay = r"F:\MAlaysia\AOI\Administration\Malaysia\States.shp"
shp_indone = r"F:\MAlaysia\AOI\Administration\Indonesia\idn_admbnda_Provinces_20200401.shp"
shp_raster_grid = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index\grid_01degree_210_496.shp"

# palm_tif ="/Volumes/PortableSSD/Malaysia/AOI/High_resolution_global_industrial_and_smallholder_oil_palm_map_for_2019/GlobalOilPalm_OP-YoP/Malaysia_Indonesia/GlobalOilPalm_OP-YoP_mosaic100m.tif"
palm_txt = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index\palm_index_shape_210_496.txt"
out_dir = r"D:\Malaysia\Validation\3_correlation\SIF"
       
in_tif_dir = r"F:\MAlaysia\SIF\GOSIF\01_tif\res_01_yearly_mean"
tifs = glob.glob(in_tif_dir + os.sep + "*.tif")    

""" prepare Yield df"""
df_malay = pd.read_csv(yield_csv_malay, index_col=0, usecols=[0,1,2,3])
df_malay = df_malay.iloc[0:12,:]#delete unwanted rows
regions_malay = df_malay.index.tolist()

df_indone = pd.read_csv(yield_csv_indone, index_col=1)
df_indone = df_indone.iloc[:,1:] #delete unwanted column
regions_indone = df_indone.index.tolist()

### convert ton/ha in Indone
year_list_indone = df_indone.columns.tolist()
year_list_indone = [y.split("_")[0] for y in year_list_indone]
year_list_indone = list(set(year_list_indone))

## From excel 
# FFB = 4.1171*CPO + 447.06
for yr in year_list_indone:
    ton_yr = df_indone[f"{yr}_ton"]
    ha_yr = df_indone[f"{yr}_ha"]
    ton_per_ha = (4.1171 * ton_yr + 447.06) / ha_yr #FFB
    df_indone[f"{yr}"] = ton_per_ha #ton/ha

df_indone = df_indone[year_list_indone]

## Combine Malaysia and Indonesia data
df_all = pd.concat([df_malay, df_indone])

## eliminate outliers by quantile
arr_all = df_all.values
arr_all = np.ravel(arr_all)
arr_all = arr_all[~np.isnan(arr_all)]

def get_quantile(ar1d):
    quantile1 = np.percentile(ar1d, 25)
    quantile2 = np.percentile(ar1d, 50)
    quantile3 = np.percentile(ar1d, 75)
    
    iqr = quantile3 - quantile1
    lowval = quantile1 - iqr*1.5
    upperval = quantile3 + iqr*1.5
    
    return lowval, upperval

## remove outliers
low_val, upper_val = get_quantile(arr_all)
df_all_clean = df_all.mask(df_all > upper_val, np.nan)
df_all_clean = df_all_clean.mask(df_all_clean < low_val, np.nan)
df_yield = df_all_clean

""" find grids within region"""
gdf_malay = gpd.read_file(shp_malay)
gdf_indone = gpd.read_file(shp_indone)
gdf_grid = gpd.read_file(shp_raster_grid)
## palm index
df_palm = pd.read_csv(palm_txt, header=None)
palm_index_list = df_palm.loc[:,0].values.tolist()

def cal_sif_mean(gdf_country, namecolumn):
    region_mean_results = {}
    for i,row in tqdm(gdf_country.iterrows()): #gdf_malay
        print("num of regions: ",len(gdf_country))
        regi_poly = row.geometry
        gdf_regi = gpd.GeoDataFrame({"geometry":[regi_poly]}).set_crs(gdf_country.crs)
        # regi_polys = gdf_regi.explode(index_parts=True) #Multipoly to polygons
        regi_name = row[namecolumn] #.NAME_1
        if regi_name =='Trengganu': #shp name wrong?
            regi_name = 'Terengganu' 
            
        
        """ # select target grids"""
        ## grids which intersect with region polygon
        gdf_tar_grid = gdf_grid[gdf_grid.intersects(regi_poly)] #index is target tif filename
        
        ### extract grid index overlap with palm index ###
        gdf_tar_grid_palm = gdf_tar_grid[gdf_tar_grid['raster_val'].isin(palm_index_list)]
        tar_grid_palm_index = gdf_tar_grid_palm.raster_val.values.tolist()
        tar_grid_palm_index = [int(t) for t in tar_grid_palm_index]
        
        """ # compute yearly sum on region palm index"""
        region_means_yr = {}
        ### Select year ###
        for yer in year_list_indone:
            tif_yr = [t for t in tifs if f"_{yer}.tif" in t][0]
            
            with rasterio.open(tif_yr) as src:
                arr = src.read(1)
                arr_1d = np.ravel(arr)
                
                ### remove outliers ###
                arr_1d_nan = arr_1d[~np.isnan(arr_1d)]
                low_val, upper_val = get_quantile(arr_1d_nan)
                arr_remove = np.where(arr_1d > upper_val, np.nan, arr_1d)
                arr_remove = np.where(arr_remove < low_val, np.nan, arr_remove)
                
                
            sif_vals = arr_remove[tar_grid_palm_index]
            sif_vals_sum = np.nanmean(sif_vals) #should be mean

            if sif_vals_sum ==0:
                sif_vals_sum = np.nan

            region_means_yr[yer] = sif_vals_sum
                       
        region_mean_results[regi_name] =  region_means_yr
    
    return region_mean_results
                   
    
region_mean_malay = cal_sif_mean(gdf_malay, "NAME_1")
region_mean_indone = cal_sif_mean(gdf_indone, "ADM1_EN")

## Finalize
region_mean_malay_fin = {}
for r,ydi in region_mean_malay.items():
    if len(ydi)>0:
        region_mean_malay_fin[r] = ydi
        
region_mean_indone_fin = {}
for r,ydi in region_mean_indone.items():
    if len(ydi)>0:
        region_mean_indone_fin[r] = ydi


""" # Comparison with Yields and palsar"""
df_region_malay = pd.DataFrame.from_dict(region_mean_malay_fin, orient='index')
df_region_indone = pd.DataFrame.from_dict(region_mean_indone_fin, orient='index')
df_region_combine = pd.concat([df_region_malay, df_region_indone])

## create data set
region_names = df_yield.index.tolist()
year_list = df_yield.columns.tolist()

## try: extract data of Malaysia / Indone -> no correlation
# df_yield_indone = df_yield.loc[regions_indone]

yield_palsar_vals = []
for yr in year_list:
    for reg in region_names:
        try:
            yield_val = df_yield_indone.at[reg, yr] #df_yield
            palsar_val = df_region_combine.at[reg, yr]
        except KeyError as e:
            yield_val, palsar_val = np.nan, np.nan
            
        yield_palsar_vals.append([yield_val, palsar_val])
            
## make dataframe
df_comparison = pd.DataFrame(yield_palsar_vals, columns=["yields","palsar"])
# delete nan
df_comparison = df_comparison.dropna(how='any')

""" # Comparison by a liner plotting"""
## regression line
mod = LinearRegression()
df_x = df_comparison[["yields"]]
df_y = df_comparison[["palsar"]]
mod_lin = mod.fit(df_x, df_y)
y_predict = mod_lin.predict(df_x)
r2_lin = mod.score(df_x, df_y)
#RMSE計算
x_val = df_comparison['yields']
y_val = df_comparison['palsar']
rmse = np.sqrt(mean_squared_error(x_val, y_val))
#Pearson Correlation Coefficient #https://realpython.com/numpy-scipy-pandas-correlation-python/#pearson-correlation-numpy-and-scipy-implementation
r, p = scipy.stats.pearsonr(x_val, y_val)

fig = plt.figure()
ax = fig.add_subplot(111, xlabel="Yield[ton/ha]", ylabel="PALSAR[dB]", title = f"SIF annual sum")

scatter = ax.scatter(x_val, y_val) #c="dodgerblue"
# plt.xlim(200,x_max)
# plt.ylim(200,y_max)
ax.text(0.7,0.1,'$ r $=' + str(round(r, 4)),fontsize=14, transform=ax.transAxes)
# ax.text(0.7,0.2,'$ RMSE $=' + str(round(rmse, 4)),fontsize=14, transform=ax.transAxes)
ax.plot(df_x, y_predict, color = 'red', linewidth=0.5)
        
fig.savefig(os.path.join(out_dir, f"correlation_SIF_mean_indone.png"), dpi=300)
plt.close()

# src_palm.close()
    



