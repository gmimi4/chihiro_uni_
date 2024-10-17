# -*- coding: utf-8 -*-
"""
Correlation with annual yield and vars with time lag
ファイルがあると出力しないので適宜直して！
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
from statistics import mean
import pingouin as pg
# os.chdir(r"C:\Users\chihiro\Desktop\Python\ANALYSIS\YieldWater")
os.chdir("/Users/wtakeuchi/Desktop/Python/ANALYSIS/YieldWater")
import _csv_to_dataframe   

# yield_csv_malay = r"D:\Malaysia\Validation\1_Yield_doc\Malaysia\Malaysia.csv"
# yield_csv_indone = r"D:\Malaysia\Validation\1_Yield_doc\Indonesia\Indonesia_CPO.csv"
# shp_region = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
# shp_extent = r"F:\MAlaysia\AOI\extent\Malaysia_and_Indonesia_extent_divided.shp"
# shp_01grid_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index"
# shp_grid = shp_01grid_dir + os.sep + "grid_01degree_210_496.shp"
# palm_txt2002 = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index\grid_01degree_210_496_palm2002.txt"
# var_csv_dir = r"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels_until2023"
# out_dir = r"D:\Malaysia\02_Timeseries\YieldWater\01_correlation_timelag\_partial"

yield_csv_malay = "/Volumes/SSD_2/Malaysia/Validation/1_Yield_doc/Malaysia/Malaysia.csv"
yield_csv_indone = "/Volumes/SSD_2/Malaysia/Validation/1_Yield_doc/Indonesia/Indonesia_CPO.csv"
shp_region = "/Volumes/SSD_2/Malaysia/Validation/1_Yield_doc/shp/region_slope_fin.shp"
shp_extent = "/Volumes/PortableSSD/Malaysia/AOI/extent/Malaysia_and_Indonesia_extent_divided.shp"
shp_01grid_dir = "/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/0_palm_index"
shp_grid = shp_01grid_dir + os.sep + "grid_01degree_210_496.shp"
palm_txt2002 = "/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/0_palm_index/grid_01degree_210_496_palm2002.txt"
var_csv_dir = "/Volumes/PortableSSD/Malaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels_until2023"
out_dir = "/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/01_correlation_timelag/_partial"    

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
def calc_peason(df_csv_): ## partial correlation
    ## it uses z score
    
    pearsonvar = {}
    for tarvar in varlist:
        # for var in varlist:
        pearson_m = {}
        pearson_m_1 = {}
        for mon in range(1,13,1): 
            df_mon_list = []
            df_mon_list_1 = []
            """ preprare dataset for partial for specific month """
            for var in varlist:
                # rows in same months
                df_csv_m = df_csv[df_csv.index.month == int(mon)]
                
                """ yield and var set for years """
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
                # collect df_month (same month)
                df_mon_list.append(df_month)
                df_mon_list_1.append(df_month_1)
                
            
            df_dataset = pd.concat(df_mon_list, axis =1)
            df_dataset_1 = pd.concat(df_mon_list_1, axis =1)
            ## extract yield column then delete
            df_dataset = df_dataset.loc[:, ~df_dataset.columns.duplicated()]
            df_dataset_1 = df_dataset_1.loc[:, ~df_dataset_1.columns.duplicated()]
            
            
            """ # Calculate the Pearson correlation coefficient between columns """            
            ### まだspecific month
            varlist_tmp = varlist.copy()
            varlist_tmp.remove(tarvar)
            varlist_tmp = [v+str(mon) for v in varlist_tmp]
            try:
                # partial_corr_matrix = pg.pcorr(df_dataset)
                partial_corr_var = pg.partial_corr(data=df_dataset, x=f"{tarvar}{mon}", y="yield", covar=varlist_tmp)
                corr = partial_corr_var.r.values[0]
                if partial_corr_var["p-val"].values[0] >0.1:
                    corr = np.nan
            except:
                corr = np.nan
        
            try:
                partial_corr_var = pg.partial_corr(data=df_dataset_1, x=f"{tarvar}{mon}", y="yield", covar=varlist_tmp)
                corr_1 = partial_corr_var.r.values[0]
                if partial_corr_var["p-val"].values[0] >0.1:
                    corr_1 = np.nan
            except:
                corr_1 = np.nan
            
            """ その月のその変数(tarvar)のcorrelation"""
            pearson_m[month_calendar[mon]] = corr #tarvarのその月のcorrelationを回収
            pearson_m_1[ f"{month_calendar[mon]}_1"] = corr_1
            
        """ # concat from t-1 to t""" #tarvarの全月の結果をまとめる
        df_peason_m = pd.DataFrame.from_dict([pearson_m]).T
        df_peason_m_1 = pd.DataFrame.from_dict([pearson_m_1]).T
        df_pearson_all = pd.concat([df_peason_m_1, df_peason_m])
        df_pearson_all.columns=[tarvar] #name column
        df_pearson_all_abs = df_pearson_all.abs()
        
        # pearsonvar[tarvar] = df_pearson_all_abs #input abs
        pearsonvar[tarvar] = df_pearson_all #with sign
        
    return pearsonvar #dict of peasons for vars in one csv



#--------------------------------
""" Process by region """
#--------------------------------
   
varlist = ['rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD'] #'GOSIF', 
month_calendar = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec" }


for i, row in tqdm(gdf_region.iterrows()):
    # i=31
    # row = gdf_region.loc[i]
    regipoly = row.geometry
    reginame = row.Name
    print(reginame)
    regioname_fin = reginame.replace(" ","")
    ## pass if no data region
    if reginame in nondata_region:
        continue
    else:
        check = out_dir + os.sep + f"partial_{regioname_fin}_stdabs.csv"
        if os.path.isfile(check):
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
            peason_pixels_vars_std = {}
            for var in varlist:
                peas_var = [pdic[var] for pdic in peason_pixels_vars] #list of peason series of var
                # df_peas_var = pd.DataFrame(peas_var).T
                df_peas_var = pd.concat(peas_var, axis=1)
                df_peas_var_mean = df_peas_var.mean(axis=1)
                df_peas_var_mean.name = var
                
                peason_pixels_vars_mean[var] = df_peas_var_mean
                
                ### std
                df_peas_var_std = df_peas_var.std(axis=1)
                df_peas_var_std.name = var
                peason_pixels_vars_std[var] = df_peas_var_std
                    
                    
                # """ # Plot by var"""
                # regioname_fin = reginame.replace(" ","")
                
                # fig = plt.figure(figsize=(12, 5))
                # fig.subplots_adjust()
                # ax = fig.add_subplot(1,1,1)
                # ax.plot(df_peas_var_mean.index, df_peas_var_mean)
                # ax.tick_params(axis='x', labelsize=14)
                # ax.tick_params(axis='y', labelsize=14)
                # plt.xticks(rotation=45)
                # ax.set_ylabel('Peason coef', fontsize = 14)
                # plt.tight_layout()
                # ### Export fig
                # out_dir_fig = out_dir + os.sep + "_png" + os.sep + var
                # os.makedirs(out_dir_fig, exist_ok=True)
                # fig.savefig(out_dir_fig + os.sep + f"partial_{regioname_fin}_{var}.png")
                # plt.close()
    
                    
            ### concat and Export csv
            df_peason_region_mean = pd.DataFrame.from_dict(peason_pixels_vars_mean)
            df_peason_region_mean.to_csv(out_dir + os.sep + f"partial_{regioname_fin}_abs.csv")
            ## std
            df_peason_region_std = pd.DataFrame.from_dict(peason_pixels_vars_std)
            df_peason_region_std.to_csv(out_dir + os.sep + f"partial_{regioname_fin}_stdabs.csv")
    
        """ # Plot"""
        # x_label = list(month_calendar.values())
        units = {'GOSIF':"W/m2/μm/sr/month", 'rain':"mm", 'temp':"degreeC", 
                  'VPD':"hPa", 'Et':"mm/day", 'Eb':"mm/day", 'SM':"m3/m3", 'VOD':""}
        
        
        fig,axes = plt.subplots(4,2, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.5)  
        for i,var in enumerate(varlist):
            row, col = divmod(i, 2)
            ax = axes[row, col]
            # ax = axes[i]
            ax.errorbar(df_peason_region_mean.index, df_peason_region_mean[var].values, 
                        yerr=df_peason_region_std[f"{var}"].values, color='blue', ecolor="lightgrey",
                        label = f"{var}",  fmt='-o', capsize=1)
            ax.tick_params(axis='y', labelsize=10)
            ax.set_ylabel(f"{var}", fontsize = 12)
            ax.legend(fontsize=14, frameon=False, loc = "upper left") #bbox_to_anchor=(.8, 0.8)
            ax.set_ylim(-1,1) #(0,1)
            # if i == len(varlist)-1:
            # if (row ==1&col==0)or(row ==2&col==1):
            # if i==6 or i==7: #諦め
            ax.tick_params(axis='x', labelsize=10)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            # else:
            #     ax.tick_params(axis='x', labelsize=0)
            if (row ==3)&(col==1):
                fig.delaxes(ax)
                # ax.set_visible(False)
                # for spine in ax.spines.values():
                #     spine.set_visible(False)
            if i ==0:
                ax.set_title(f"{reginame} partial correlaton")
            # fig.delaxes(axes[3,1])
        axes[3, 1].set_axis_off()
        plt.tight_layout()
        ### Export fig
        out_dir_fig = out_dir + os.sep + "_png"
        os.makedirs(out_dir_fig, exist_ok=True)
        fig.savefig(out_dir_fig + os.sep + f"{regioname_fin}_partial.png")
        plt.close()
        
        
# """ # ミスってplot from csv"""
csv_dir = "/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/01_correlation_timelag/_partial"
csvs = glob.glob(csv_dir + os.sep + "*_abs.csv")
csvs_std = glob.glob(csv_dir + os.sep + "*_stdabs.csv")

varlist = ['rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD'] #'GOSIF', 

for csvf in csvs:
    reginame = os.path.basename(csvf)[:-4].split("_")[1]
    std_file = [f for f in csvs_std if reginame in f][0]
    df_mean = pd.read_csv(csvf, index_col=0)
    df_std = pd.read_csv(std_file, index_col=0)
    
    fig,axes = plt.subplots(4,2, figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5)  
    for i,var in enumerate(varlist):
        row, col = divmod(i, 2)
        ax = axes[row, col]
        # ax = axes[i]
        ax.errorbar(df_mean.index, df_mean[var].values, 
                    yerr=df_std[f"{var}"].values, color='blue', ecolor="lightgrey",
                    fmt='-o', capsize=1) #label = f"{var}",
        ax.tick_params(axis='y', labelsize=10)
        ax.set_ylabel(f"{var}", fontsize = 12)
        # ax.legend(fontsize=14, frameon=False, loc = "upper left") #bbox_to_anchor=(.8, 0.8)
        ax.set_ylim(-1,1)
        ax.axhline(y=0,color='grey', linewidth=0.7)
        ax.tick_params(axis='x', labelsize=10)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        if (row ==3)&(col==1):
            fig.delaxes(ax)
        if i ==0:
            ax.set_title(f"{reginame} partial correlaton")
        # fig.delaxes(axes[3,1])
    axes[3, 1].set_axis_off()
    plt.tight_layout()
    ### Export fig
    out_dir_fig = out_dir + os.sep + "_png"
    os.makedirs(out_dir_fig, exist_ok=True)
    fig.savefig(out_dir_fig + os.sep + f"{reginame}_partial.png")
    plt.close()
    
    

