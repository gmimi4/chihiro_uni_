# -*- coding: utf-8 -*-
"""
Correlation with annual yield and vars with time lag
"""
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
# from rasterstats import zonal_stats
from tqdm import tqdm
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
import scipy.stats
from scipy.stats import zscore
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import glob
import numpy as np
from statistics import mean
import math
os.chdir(r"C:\Users\chihiro\Desktop\Python\ANALYSIS\YieldWater")
import _csv_to_dataframe   

yield_csv_malay = r"D:\Malaysia\Validation\1_Yield_doc\Malaysia\Malaysia.csv"
yield_csv_indone = r"D:\Malaysia\Validation\1_Yield_doc\Indonesia\Indonesia_CPO.csv"
shp_region = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
shp_extent = r"F:\MAlaysia\AOI\extent\Malaysia_and_Indonesia_extent_divided.shp"
shp_01grid_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index"
shp_grid = shp_01grid_dir + os.sep + "grid_01degree_210_496.shp"
palm_txt2002 = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index\grid_01degree_210_496_palm2002.txt"
var_csv_dir = r"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels_until2023"
out_dir = r"D:\Malaysia\02_Timeseries\YieldWater\01_correlation_timelag"
    

""" prepare Yield df"""
df_malay = pd.read_csv(yield_csv_malay, index_col=0)
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
year_list_indone = sorted(list(set(year_list_indone)))

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

""" z scoring yield"""
df_yield_z = df_yield.copy() 
for i, row in df_yield.iterrows():
    row_valid = row[~np.isnan(row)]
    reginame = row_valid.name
    years_row = row_valid.index.tolist()
    row_valid_z = zscore(row_valid, ddof=1) #ddof=0 標準偏差、ddof=1 不偏標準偏差
    for y in years_row:
        zs = row_valid_z.at[y]
        df_yield_z.at[reginame,y] = zs


nondata_region = list(df_yield_z[df_yield_z.isna().all(axis=1)].index)

""" set year list"""
year_list = [y for y in range(2002,2024,1)]

""" set gdf """
gdf_region = gpd.read_file(shp_region)
gdf_grid = gpd.read_file(shp_grid)
gdf_extent = gpd.read_file(shp_extent)

gdf_A1 = gpd.read_file(shp_01grid_dir + os.sep + f"grid_01degree_A1.shp")
gdf_A2 = gpd.read_file(shp_01grid_dir + os.sep + f"grid_01degree_A2.shp")
gdf_A3 = gpd.read_file(shp_01grid_dir + os.sep + f"grid_01degree_A3.shp")
gdf_A4 = gpd.read_file(shp_01grid_dir + os.sep + f"grid_01degree_A4.shp")
gdf_A_dic = {"A1":gdf_A1, "A2":gdf_A2, "A3":gdf_A3, "A4":gdf_A4}

""" set palm """
df_palm = pd.read_csv(palm_txt2002, header=None)
list_palm = df_palm[0].values.tolist()


""" # def to extract target A* csv file"""
def find_page(gdfpoi):
    for i,row in gdf_extent.iterrows():
        grid = row.geometry
        if gdfpoi.within(grid).values[0]: #if point is on the line, it's false
            page = row.PageName
        else:
            if gdfpoi.intersects(grid).values[0]: #if point is on the line, it's false
                page = row.PageName
    return page

def find_index(gdfpoi, pagenum):
    gdf_page = gdf_A_dic[pagenum]
    gdf_page_intersecting = gdf_page[gdf_page.intersects(gdfpoi.geometry.values[0])]
    index_want = gdf_page_intersecting.raster_val.values[0]
    return int(index_want)

           
def find_csv(regi_poly):
    # regi_poly = gdf_region.loc[31].geometry
    
    gdf_regi = gpd.GeoDataFrame({"geometry":[regi_poly]}).set_crs(gdf_region.crs) #multipolygon
    
    """ # select target grids"""
    ## grids which intersect with region polygon
    gdf_tar_grid = gdf_grid[gdf_grid.intersects(regi_poly)] #index is target tif filename
    ### select grid id within palm 2
    gdf_tar_grid = gdf_tar_grid[gdf_tar_grid['raster_val'].isin(list_palm)]
    
    ## convert to point
    gdf_tar_grid['centroid'] = gdf_tar_grid.geometry.centroid
    gdf_centroids = gdf_tar_grid.copy()
    gdf_centroids['geometry'] = gdf_centroids['centroid']
    gdf_centroids = gdf_centroids.drop(columns=['centroid'])
    
    csvlist = []
    for poi in gdf_centroids.geometry:
        gdfp = gpd.GeoDataFrame({"geometry":[poi]}).set_crs(gdf_region.crs)
        A = find_page(gdfp)
        index_target = find_index(gdfp, A)
        csvfile = var_csv_dir + os.sep + A + os.sep + f"{index_target}.csv"
        csvlist.append(csvfile)
    
    return csvlist
        
""" # def peason for a single csvfile"""
def calc_peason(df_csv_):
    pearsonvar = {}
    for var in varlist:    
        pearson_m = {}
        pearson_m_1 = {}
        for mon in range(1,13,1):
            # rows in same months
            df_csv_m = df_csv[df_csv.index.month == int(mon)]
            
            yield_var_m = {} #list for a specific month for peason
            yield_var_m_1 = {}
            # extract vars and yield in year t
            for yr in years_region:
                yrt = int(yr)
                yrt_1 = int(yrt) - 1            
                df_csv_m_yrt = df_csv_m[df_csv_m.index.year == yrt] #Year t
                df_csv_m_yrt_1 = df_csv_m[df_csv_m.index.year == yrt_1] #Year t-1
                ## yield at Year,t
                yield_yrt = df_yield_region.at[yr] #This is Yield at year t
                var_yrt = df_csv_m_yrt[f"{var}z"].values[0] # This is X at year t, month m
                var_yrt_1 = df_csv_m_yrt_1[f"{var}z"].values[0] # This is X at year t-1, month m
                yield_var_m[yrt]=[yield_yrt, var_yrt] #全年のspecific monthの組み合わせを作る　これをピアソン
                yield_var_m_1[yrt] = [yield_yrt, var_yrt_1]
            
            df_month = pd.DataFrame(yield_var_m).T #columns=["yield",f"{var}{mon}"]
            df_month_1 = pd.DataFrame(yield_var_m_1).T #columns=["yield",f"{var}{mon}_1"]
            # rename
            df_month = df_month.rename(columns={0:"yield",1:f"{var}{mon}"})
            df_month_1 = df_month_1.rename(columns={0:"yield",1:f"{var}{mon}"})
            # dropna
            df_month = df_month.dropna()
            df_month_1 = df_month_1.dropna()
            
            """ # Calculate the Pearson correlation coefficient between columns """
            # correlation_matrix = df_month.corr(method='pearson').iat[0,1]
            # correlation_matrix_1 = df_month_1.corr(method='pearson').iat[0,1]  
            # pearson_m[month_calendar[mon]] = correlation_matrix
            # pearson_m_1[ f"{month_calendar[mon]}_1"] = correlation_matrix_1
            
            try:
                corr, pval = pearsonr(df_month['yield'], df_month[f"{var}{mon}"])
                if pval >0.1:
                    corr = np.nan
            except:
                corr = np.nan
            
            try:
                corr_1, pval_1 = pearsonr(df_month_1['yield'], df_month_1[f"{var}{mon}"])
                if pval_1 >0.1:
                    corr_1 = np.nan
            except:
                corr_1 = np.nan
            
            pearson_m[month_calendar[mon]] = corr
            pearson_m_1[ f"{month_calendar[mon]}_1"] = corr_1
                
        """ # concat from t-1 to t"""
        df_peason_m = pd.DataFrame.from_dict([pearson_m]).T
        df_peason_m_1 = pd.DataFrame.from_dict([pearson_m_1]).T
        df_pearson_all = pd.concat([df_peason_m_1, df_peason_m])
        df_pearson_all.columns=[var] #name column
        df_pearson_all_abs = df_pearson_all.abs()
        pearsonvar[var] = df_pearson_all_abs[var] #input abs
        
    return pearsonvar #dict of peasons for vars in one csv



#--------------------------------
""" Process by region """
#--------------------------------
   
varlist = ['GOSIF', 'rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD']
month_calendar = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec" }


for i, row in tqdm(gdf_region.iterrows()):
    # i=31
    # row = gdf_region.loc[i]
    regipoly = row.geometry
    reginame = row.Name
    ## pass if no data region
    if reginame in nondata_region:
        continue
    else:
        
        """ get annual Yield"""
        df_yield_region = df_yield_z.loc[reginame,:]
        df_yield_region = df_yield_region.dropna()
        years_region = df_yield_region.index.tolist()
        
        csv_list = find_csv(regipoly)
        """ iterrate csvfile """
        peason_pixels_vars = []
        for csvfi in csv_list:
            df_csv = _csv_to_dataframe.main(csvfi)
            
            ## create peason timeseris for vars
            pearson_var = calc_peason(df_csv) ## abs
            ## collect
            peason_pixels_vars.append(pearson_var)
            
        """ #get average peason for var"""
        peason_pixels_vars_mean = {}
        for var in varlist:
            peas_var = [pdic[var] for pdic in peason_pixels_vars] #list of peason series of var
            df_peas_var = pd.DataFrame(peas_var).T
            df_peas_var_mean = df_peas_var.mean(axis=1)
            df_peas_var_mean.name = var
            
            peason_pixels_vars_mean[var] = df_peas_var_mean
                
                
            """ # Plot by var"""
            regioname_fin = reginame.replace(" ","")
            
            fig = plt.figure(figsize=(12, 5))
            fig.subplots_adjust()
            ax = fig.add_subplot(1,1,1)
            ax.plot(df_peas_var_mean.index, df_peas_var_mean)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            plt.xticks(rotation=45)
            ax.set_ylabel('Peason coef', fontsize = 14)
            plt.tight_layout()
            ### Export fig
            out_dir_fig = out_dir + os.sep + "_png" + os.sep + var
            os.makedirs(out_dir_fig, exist_ok=True)
            fig.savefig(out_dir_fig + os.sep + f"peason_{regioname_fin}_{var}.png")
            plt.close()

                
        ### concat and Export csv
        df_peason_region_mean = pd.DataFrame.from_dict(peason_pixels_vars_mean)
        df_peason_region_mean.to_csv(out_dir + os.sep + f"peason_{regioname_fin}_abs.csv")
        
