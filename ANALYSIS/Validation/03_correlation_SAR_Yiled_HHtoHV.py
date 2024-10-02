# -*- coding: utf-8 -*-
"""
Compare Yield data and SAR back scattered intensity by provinces or states
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
shp_raster_grid = r"D:\Malaysia\Validation\2_PALSAR_mosaic\_preparation\target_grid_PALSAR_1deg.shp"
palm_tif =r"F:\MAlaysia\AOI\High_resolution_global_industrial_and_smallholder_oil_palm_map_for_2019\GlobalOilPalm_OP-YoP\Malaysia_Indonesia\\GlobalOilPalm_OP-YoP_mosaic100m.tif"
out_dir = r"D:\Malaysia\Validation\3_correlation"
       
# in_tif_dir = r"D:\Malaysia\Validation\2_PALSAR_mosaic\01_res100m"
in_tif_dir = r"D:\Malaysia\Validation\2_PALSAR_mosaic\02_HHtoHV_res100m"
band = "HHtoHV"
tifs = glob.glob(in_tif_dir + os.sep + f"*{band}*.tif")


""" prepare Yield df"""
df_malay = pd.read_csv(yield_csv_malay, index_col=0, usecols=[0,1,2,3])
df_malay = df_malay.iloc[0:12,:]#delete unwanted rows
df_indone = pd.read_csv(yield_csv_indone) #index_col=1
df_indone = df_indone.iloc[:,1:] #delete unwanted column
# rename 2 Nusa Tenggara
df_indone.iat[17,0] = "Nusa Tenggara Barat"
df_indone.iat[18,0] = "Nusa Tenggara Timur"
# set index
df_indone = df_indone.set_index('Unnamed: 1')
# rename Bangka Belitung
df_indone = df_indone.rename(index={'Bangka Belitung': 'Kepulauan Bangka Belitung'})

# regions_indone = df_indone.index.tolist()

### convert ton/ha in Indone
year_list_indone = df_indone.columns.tolist()
year_list_indone = [y.split("_")[0] for y in year_list_indone]
year_list_indone = list(set(year_list_indone))

## From excel 
# FFB = 4.1171*CPO + 447.06
for yr in year_list_indone:
    ton_yr = df_indone[f"{yr}_ton"]
    ha_yr = df_indone[f"{yr}_ha"]
    ton_per_ha = (4.1171 * ton_yr) / ha_yr #FFB
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

""" prepare PALSAR in shp"""
gdf_malay = gpd.read_file(shp_malay)
gdf_indone = gpd.read_file(shp_indone)
gdf_grid = gpd.read_file(shp_raster_grid)
## open palm tif
src_palm = rasterio.open(palm_tif)

def cal_palsar_mean(gdf_country, namecolumn):
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
       
        ## select tif from grid id
        tar_name = gdf_tar_grid.index.tolist()
        tar_name = [f"_{str(t)}_res100.tif" for t in tar_name]
        tar_tifs = [t for t in tifs if "_" + os.path.basename(t).split("_")[-2] + "_res100.tif" in tar_name]
        
        
        region_means_yr = {}
        ### Select year ###
        for yer in year_list_indone:
            tar_tifs_yr = [t for t in tar_tifs if f"_{yer}_" in t]
    
            # region_grid_means = []
            region_grid_arr = []
            for tat in tar_tifs_yr: #same year in a polygon
                use_grid_id = os.path.basename(tat)[:-4].split("_")[-2]
                use_grid = gdf_grid.loc[int(use_grid_id),:]
                gdf_use_grid = gpd.GeoDataFrame({"geometry":[use_grid.geometry]}).set_crs(gdf_grid.crs)
            
                # """ # just for check: mask palm raster by regions"""
                # try: #error if regi shape not fully overlap with palm raster (only Aceh?)
                #     mask_palm, transform_palm = rasterio.mask.mask(src_palm, [regi_poly], crop=True) #iterable poly
                #     ## check if no palm exists (only 0 or 65535)
                #     mask_palm_check = np.where((mask_palm != 0) &(mask_palm != 65535))
                #     mask_palm_check = np.ravel(mask_palm_check)
                # except:
                #     continue #go to next grid tif

                # if len(mask_palm_check) ==0:
                #     continue #go to next grid tif
                # else:
                """ # mask rasters by region"""
                ### clip region by grid ###
                gdf_regi_clip = gpd.clip(gdf_regi, gdf_use_grid)
                
                
                """ # mask rasters by clipped region in a gird"""
                ## mask palsar and palm by grid (want raster index) #need same shape
                with rasterio.open(tat) as src: #palsar
                    # for outliers
                    arr_pal_ori = src.read(1)
                    arr_pal_ori_1d = np.ravel(arr_pal_ori)
                    arr_pal_ori_1d = arr_pal_ori_1d[~np.isnan(arr_pal_ori_1d)]
                    lowlim, uplim = get_quantile(arr_pal_ori_1d)
                    
                    ### mask palsar by 
                    mask_palsar, transform_palsar = rasterio.mask.mask(src, [gdf_regi_clip.geometry[0]], crop=True) #[use_grid.geometry]
                    ## remove outlier
                    mask_remove = np.where(mask_palsar > uplim, np.nan, mask_palsar)
                    mask_remove = np.where(mask_remove < lowlim, np.nan, mask_remove)
                    mask_palsar = mask_remove
                    
                # mask palm by grid
                mask_palm2, transform_palm2 = rasterio.mask.mask(src_palm, [gdf_regi_clip.geometry[0]], crop=True) #[use_grid.geometry]
                
                ### allign the smallest shape
                smallest_shape = (min([mask_palsar.shape[1],mask_palm2.shape[1]]), min([mask_palsar.shape[2],mask_palm2.shape[2]]))
                diff_shape_palsar = [mask_palsar.shape[1]-smallest_shape[0], mask_palsar.shape[2]-smallest_shape[1]]
                diff_shape_palm = [mask_palm2.shape[1]-smallest_shape[0], mask_palm2.shape[2]-smallest_shape[1]]
                
                def allign_shape(arr, shapelist):
                    if sum(shapelist)>0:
                        arr_del = np.delete(arr, slice(0, shapelist[0]), 0)
                        arr_del = np.delete(arr_del, slice(0, shapelist[1]), 1)
                    else:
                        arr_del = arr
                    return arr_del
                
                mask_palsar_cut = allign_shape(mask_palsar[0], diff_shape_palsar)
                mask_palm_cut = allign_shape(mask_palm2[0], diff_shape_palm)
                        
                            
                """ ## collect palm valid index #above planting year """
                palm2_index = np.where((mask_palm_cut !=0)&(mask_palm_cut !=65535)&(mask_palm_cut <int(yer))) #
                
                ## collect palsar vals on palms
                mask_palsar_palm = mask_palsar_cut[palm2_index]
                mask_palsar_palm = np.where(mask_palsar_palm ==0,np.nan, mask_palsar_palm)

                # ## calc mean
                # mask_palsar_palm_gamma = np.where(mask_palsar_palm_gamma !=-np.inf, mask_palsar_palm_gamma, np.nan)
                # mask_palsar_mean = np.nanmean(mask_palsar_palm_gamma) # meam of grid*
                
                region_grid_arr.append(mask_palsar_palm) #array in grid* in year*
                
                
            ### calc mean by year ###
            region_grid_arr_flat = [value for arr in region_grid_arr for value in arr]
            region_grid_means = [n for n in region_grid_arr_flat if not np.isnan(n)]
            if len(region_grid_means)>0:
                yr_mean = statistics.mean(region_grid_means) #mean of all grids in a year*
            else:
                yr_mean = np.nan
            
            # yr_mean = 10*math.log10((yr_mean**2)) -83.0# γ₀ = 10log₁₀(DN²) - 83.0 dB
            # add year mean of the year to dic     
            region_means_yr[yer] = yr_mean
                       
        region_mean_results[regi_name] =  region_means_yr
    
    return region_mean_results
                   
    
region_mean_malay = cal_palsar_mean(gdf_malay, "NAME_1")
region_mean_indone = cal_palsar_mean(gdf_indone, "ADM1_EN")

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

yield_palsar_vals = []
for yr in year_list:
    for reg in region_names:
        try:
            yield_val = df_yield.at[reg, yr]
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
ax = fig.add_subplot(111, xlabel="Yield[ton/ha]", ylabel="PALSAR[dB]", title = f"{band}")

scatter = ax.scatter(x_val, y_val) #c="dodgerblue"
# plt.xlim(200,x_max)
# plt.ylim(200,y_max)
ax.text(0.7,0.1,'$ r $=' + str(round(r, 4)),fontsize=14, transform=ax.transAxes)
# ax.text(0.7,0.2,'$ RMSE $=' + str(round(rmse, 4)),fontsize=14, transform=ax.transAxes)
ax.plot(df_x, y_predict, color = 'red', linewidth=0.5)
        
fig.savefig(os.path.join(out_dir, f"correlation_{band}.png"), dpi=300)
plt.close()

src_palm.close()
    



