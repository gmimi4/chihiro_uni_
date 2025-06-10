# -*- coding: utf-8 -*-
"""
Correlation with annual yield and vars with time lag
Collect pearson as 0 if p not significant
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
# os.chdir('/Users/wtakeuchi/Desktop/Python/ANALYSIS/YieldWater')
import _csv_to_dataframe
import _yield_csv

pp = "_regionalmean"
yield_csv_malay = r"D:\Malaysia\Validation\1_Yield_doc\Malaysia\Malaysia.csv"
yield_csv_indone = r"D:\Malaysia\Validation\1_Yield_doc\Indonesia\Indonesia_CPO.csv"
shp_region = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
shp_extent = r"F:\MAlaysia\AOI\extent\Malaysia_and_Indonesia_extent_divided.shp"
shp_01grid_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index_EVI"
shp_grid = shp_01grid_dir + os.sep + "grid_01degree_210_491.shp"
palm_txt2002 = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index\grid_01degree_210_491.txt" #all palm
out_dir = rf"D:\Malaysia\02_Timeseries\YieldWater\01_correlation_timelag\{pp}"
os.makedirs(out_dir, exist_ok=True)

mean_dir = r"D:\Malaysia\02_Timeseries\CCM\01_region_mean\EVI_allage"

# yield_csv_malay = "/Volumes/SSD_2/Malaysia/Validation/1_Yield_doc/Malaysia/Malaysia.csv"
# yield_csv_indone = "/Volumes/SSD_2/Malaysia/Validation/1_Yield_doc/Indonesia/Indonesia_CPO.csv"
# shp_region = "/Volumes/SSD_2/Malaysia/Validation/1_Yield_doc/shp/region_slope_fin.shp"
# shp_extent = "/Volumes/PortableSSD/Malaysia/AOI/extent/Malaysia_and_Indonesia_extent_divided.shp"
# shp_01grid_dir = "/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/0_palm_index"
# shp_grid = shp_01grid_dir + os.sep + "grid_01degree_210_496.shp"
# palm_txt2002 = "/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/0_palm_index/grid_01degree_210_496_palm2002.txt"
# var_csv_dir = "/Volumes/PortableSSD/Malaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels_until2023"
# out_dir = f"/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/01_correlation_timelag/{pp}"
# os.makedirs(out_dir, exist_ok=True)
    

""" prepare Yield df"""
df_yield, df_yield_z, df_yield_detr = _yield_csv.main()
nondata_region = list(df_yield_z[df_yield_z.isna().all(axis=1)].index)
nondata_region = nondata_region + ["Gorontalo"]

""" set year list"""
year_list = [y for y in range(2002,2024,1)]

""" set gdf """
gdf_region = gpd.read_file(shp_region)

        
""" # def peason for a single csvfile"""
def calc_peason(df_csv_):
    # df_csv_ = df_csv_z
    pearsonvar = {}
    for var in varlist:    
        pearson_m = {}
        pearson_m_1 = {}
        for mon in range(1,13,1):
            # rows in same months
            df_csv_m = df_csv_[df_csv_.index.month == int(mon)]
            
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
                    # corr = np.nan
                    corr = 0
            except:
                corr = np.nan
            
            try:
                corr_1, pval_1 = pearsonr(df_month_1['yield'], df_month_1[f"{var}{mon}"])
                if pval_1 >0.1:
                    # corr_1 = np.nan
                    corr_1=0
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
        pearsonvar[var] = df_pearson_all[var]
        
    return pearsonvar #dict of peasons for vars in one csv



#--------------------------------
""" Process by region """
#--------------------------------
   
varlist = ['EVI', 'rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD']
month_calendar = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec" }
units = {'EVI':"", 'rain':"mm", 'temp':"degreeC", 
          'VPD':"hPa", 'Et':"mm/day", 'Eb':"mm/day", 'SM':"m3/m3", 'VOD':""}

csvs = glob.glob(mean_dir + os.sep + "*.csv")

for i, row in tqdm(gdf_region.iterrows()):
    # i=37
    # row = gdf_region.loc[i]
    regipoly = row.geometry
    reginame = row.Name
    regioname_fin = reginame.replace(" ","")
    ## pass if no data region
    if reginame in nondata_region:
        continue
    else:
        
        """ get annual Yield"""
        df_yield_region = df_yield_detr.loc[reginame,:]
        df_yield_region = df_yield_region.dropna()
        years_region = df_yield_region.index.tolist()
        
        csvfi = [c for c in csvs if f"{regioname_fin}_meanval" in os.path.basename(c)][0]
        """ iterrate csvfile """
        peason_pixels_vars = []
        df_csv = pd.read_csv(csvfi, index_col=0, parse_dates=True)
        
        """ # zscoring with all seson"""         
        df_csv_z = df_csv.copy()
        for var in varlist:
            df_csv_z[f"{var}z"] = np.nan
            df_var = df_csv_z.loc[:,var]
            var_mean = df_var.mean()
            var_std = df_var.std()            
            df_csv_z[f"{var}z"] = (df_csv_z[var]-var_mean)/var_std
        
        """ # Scaling 0-1""" ##
        for var in varlist:
            df_csv_z[f"{var}s"] = np.nan
            var_max = df_csv_z[var].max()
            var_min = df_csv_z[var].min()
            df_csv_z[f"{var}s"] = (df_csv_z[var]-var_min)/(var_max - var_min)
        
        ## create peason timeseris for vars
        pearson_var = calc_peason(df_csv_z) ## abs
            
        ### concat and Export csv
        df_peason_region_mean = pd.DataFrame.from_dict(pearson_var)
        df_peason_region_mean.to_csv(out_dir + os.sep + f"peason_{regioname_fin}_abs.csv")
    
        ## tmp for plot ###
        df_peason_region_std = df_peason_region_mean.copy()
        df_peason_region_std.loc[:,:]=0
        
        """ # Plot by var"""
        
        # fig,axes = plt.subplots(4,2, figsize=(8, 8))
        # fig.subplots_adjust(hspace=0.5)  
        # for i,var in enumerate(varlist):
        #     row, col = divmod(i, 2)
        #     ax = axes[row, col]
        #     # ax = axes[i]
        #     ax.errorbar(df_peason_region_mean.index, df_peason_region_mean[var].values, 
        #                 yerr=df_peason_region_std[f"{var}"].values, color='blue', ecolor="lightgrey",
        #                 label = f"{var}",  fmt='-o', capsize=1)
        #     ax.tick_params(axis='y', labelsize=10)
        #     ax.set_ylabel(f"Pearson_{var}", fontsize = 16)
        #     ax.legend(fontsize=14, frameon=False, loc = "upper left") #bbox_to_anchor=(.8, 0.8)
        #     ax.set_ylim(-1,1) #(0,1)
        #     # if i == len(varlist)-1:
        #     # if (row ==1&col==0)or(row ==2&col==1):
        #     # if i==6 or i==7: #諦め
        #     ax.tick_params(axis='x', labelsize=10)
        #     plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        #     # else:
        #     #     ax.tick_params(axis='x', labelsize=0)
        #     if (row ==3)&(col==1):
        #         fig.delaxes(ax)
        #         # ax.set_visible(False)
        #         # for spine in ax.spines.values():
        #         #     spine.set_visible(False)
        #     if i ==0:
        #         ax.set_title(f"{reginame} pearson correlaton")
        #     # fig.delaxes(axes[3,1])
        # axes[3, 1].set_axis_off()
        # plt.tight_layout()
        # ### Export fig
        # out_dir_fig = out_dir + os.sep + "_png"
        # os.makedirs(out_dir_fig, exist_ok=True)
        # fig.savefig(out_dir_fig + os.sep + f"{regioname_fin}{pp}.png")
        # plt.close()
        

        
""" # from CSV"""
# csv_dir = f"/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/01_correlation_timelag/{pp}"
csv_dir = rf"D:\Malaysia\02_Timeseries\YieldWater\01_correlation_timelag\{pp}"
csvs = glob.glob(csv_dir + os.sep + "*_abs.csv")

varlist = ['rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD'] #'GOSIF', 
newvarlist = {'rain':"Rain", 'temp':"Temp", 'VPD':"VPD", 'Et':"Trans", 'Eb':"Evapo", 'SM':"SM", 'VOD':"VOD"}


for csvf in csvs:
    reginame = os.path.basename(csvf)[:-4].split("_")[1]
    df_mean = pd.read_csv(csvf, index_col=0)
    fig,axes = plt.subplots(7,1, figsize=(10, 28))
    fig.subplots_adjust(hspace=0.5)  
    for i, var in enumerate(varlist):
        ax = axes[i]
        # ax.errorbar(df_mean.index, df_mean[var].values, 
        #             yerr=df_std[f"{var}"].values, color='blue', ecolor="grey",
        #             fmt='-o', capsize=1)
        ax.plot(df_mean.index, df_mean[var].values, 
                color='blue', marker='o')
        ax.tick_params(axis='y', labelsize=14)
        varnew = newvarlist[var]
        ax.set_ylabel(f"Pearson_\n{varnew}", fontsize=16)
        ax.set_ylim(-1, 1)
        ax.axhline(y=0, color='grey', linewidth=0.7)
        ax.tick_params(axis='x', labelsize=14)
        # Hide x-axis ticks and labels except for the bottom subplot
        if i < len(varlist) - 1:
            ax.tick_params(labelbottom=False)
        else:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    out_dir_fig = out_dir + os.sep + "_png"
    os.makedirs(out_dir_fig, exist_ok=True)
    fig.savefig(out_dir_fig + os.sep + f"{reginame}{pp}.png")
    plt.close()