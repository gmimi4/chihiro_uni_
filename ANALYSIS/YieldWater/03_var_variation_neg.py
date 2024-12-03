# -*- coding: utf-8 -*-
"""
Fluctuation of an influential var at the specific month for all periods
Find negative correlation
"""
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import glob
from statsmodels.tsa.seasonal import STL
# from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from matplotlib import colorbar, colors
plt.rcParams['font.family'] = 'Times New Roman'
os.chdir(r"C:\Users\chihiro\Desktop\Python\ANALYSIS\YieldWater")
# os.chdir("/Users/wtakeuchi/Desktop/Python/ANALYSIS/YieldWater")
import _yield_csv  
import _csv_to_dataframe 
import _ADF_MK

pp = "_pearson_detr"
pearson_dir = rf"D:\Malaysia\02_Timeseries\YieldWater\01_correlation_timelag\{pp}"
shp_region = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
shp_extent = r"F:\MAlaysia\AOI\extent\Malaysia_and_Indonesia_extent_divided.shp"
shp_01grid_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index"
shp_grid = shp_01grid_dir + os.sep + "grid_01degree_210_496.shp"
palm_txt2002 = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index\grid_01degree_210_496_palm2002.txt"
var_csv_dir = r"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels_until2023"
pp2 = pp + "_neg"
out_dir = rf"D:\Malaysia\02_Timeseries\YieldWater\03_var_variation\{pp2}"

# pearson_dir = "/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/01_correlation_timelag/_partial"
# shp_region = "/Volumes/SSD_2/Malaysia/Validation/1_Yield_doc/shp/region_slope_fin.shp"
# shp_extent = "/Volumes/PortableSSD/Malaysia/AOI/extent/Malaysia_and_Indonesia_extent_divided.shp"
# shp_01grid_dir = "/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/0_palm_index"
# shp_grid = shp_01grid_dir + os.sep + "grid_01degree_210_496.shp"
# palm_txt2002 = "/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/0_palm_index/grid_01degree_210_496_palm2002.txt"
# var_csv_dir = "/Volumes/PortableSSD/Malaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels_until2023"
# out_dir = '/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/03_var_variation/_partial'

month_calendar = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec" }

varlist = ['rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD'] #'GOSIF', 
units = {'GOSIF':"W/m2/μm/sr/month", 'rain':"mm", 'temp':"degreeC", 
         'VPD':"hPa", 'Et':"mm/day", 'Eb':"mm/day", 'SM':"m3/m3", 'VOD':""}

""" FFB yield df"""
df_yield, df_yield_z, df_yield_detr = _yield_csv.main()
nondata_region = list(df_yield_z[df_yield_z.isna().all(axis=1)].index)
nondata_region = nondata_region +["Maluku Utara"]


def get_key(my_dict, val):
    for key, value in my_dict.items():
        if value == val:
            return key

""" prepare polygon"""
gdf_region = gpd.read_file(shp_region)
gdf_region = gdf_region.drop(columns=['NAME_1','ADM1_EN','rank_yield']) #'slope',
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
    # regi_poly = gdf_region.loc[0,:].geometry
    
    gdf_regi = gpd.GeoDataFrame({"geometry":[regi_poly]}).set_crs(gdf_region.crs) #multipolygon
    
    """ # select target grids"""
    ## grids which intersect with region polygon
    gdf_tar_grid = gdf_grid[gdf_grid.intersects(regi_poly)] #index is target tif filename
    ### select grid id within palm 2
    gdf_tar_grid = gdf_tar_grid[gdf_tar_grid['raster_val'].isin(list_palm)] #palm 2002
    
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
    
    
## --------------------------
""" # Process """
## --------------------------
""" find hihest pearson var and month whish are STRESSING to yield"""
pearson_csvs = glob.glob(pearson_dir + os.sep + "*_abs.csv")

criti_result = {}
for i, row in tqdm(gdf_region.iterrows()):
    # i=20
    # row = gdf_region.loc[i,:]
    regipoly = row.geometry
    reginame = row.Name
    regioname_fin = reginame.replace(" ","")
    ## pass if no data region
    if reginame in nondata_region:
        continue
    else:
        reginame_csv = reginame.replace(" ", "")
        csv_pearson = [c for c in pearson_csvs if f"peason_{reginame_csv}_abs.csv" in c][0]
        """ find hihest pearson var and mont"""
        df_peason = pd.read_csv(csv_pearson, index_col=0)
        # df_peason_abs = df_peason.abs()
        ## drop GOSIF
        # df_peason = df_peason.drop("GOSIF", axis=1)
        
        if len(df_peason.dropna(how='all')) == 0:
            criti_result[reginame] = ["", 0, np.nan, np.nan]
        else:
            ### Remain only negative correlation
            df_peason_neg = df_peason.applymap(lambda x: np.nan if x > 0 else x)
            df_peason_abs = df_peason_neg.abs()

            df_peason_abs = df_peason_abs.drop(["GOSIF"],axis=1)
            ### Find critical var name and month
            criti_var = df_peason_abs.max().idxmax()
            criti_month = df_peason_abs[criti_var].idxmax() #month str
            criti_month = get_key(month_calendar, criti_month.split('_')[0])
            
            
            """ Extract timeseries data of critical var at critical month"""
            csv_list = find_csv(regipoly)
            
            df_csv_list = []
            df_csv_list_s = []
            for csvfi in csv_list:
                df_csv = _csv_to_dataframe.main(csvfi)
                
                ### collect df_csv for averaging by var            
                ### ori data
                df_var = df_csv[criti_var] #ori data
                df_var_month = df_var[df_var.index.month==criti_month]
                df_var_month = df_var_month[~np.isnan(df_var_month)]
                df_csv_list.append(df_var_month)
                
                ### scaled data -> scaling for specific month
                # df_var_s = df_csv[criti_var+"s"] #scaled data for std and slope
                # df_var_month_s = df_var_s[df_var_s.index.month==criti_month]
                # df_var_month_s = df_var_month_s[~np.isnan(df_var_month_s)]
                df_var_month_s  = (df_var_month - df_var_month.min()) / (df_var_month.max() - df_var_month.min())
                df_csv_list_s.append(df_var_month_s)
            
            """ averaging timeseries data"""
            df_criti = pd.concat(df_csv_list,axis=1)
            df_criti_ave = df_criti.mean(axis=1)
            df_criti_std = df_criti.std(axis=1)
            
            df_criti_s = pd.concat(df_csv_list_s,axis=1)
            df_criti_ave_s = df_criti_s.mean(axis=1) #use this
            df_criti_std_s = df_criti_s.std(axis=1)
            
            ## CV --> Std
            criti_sd = df_criti_ave_s.std()
            # criti_mean = df_criti_ave_s.mean()
            # criti_cv = criti_sd/criti_mean
            
            ### deseasonality ###
            # stl_result = STL(df_criti_ave_s, period=12).fit()
            # stl_seasonal = stl_result.seasonal
            # stl_deseason = stl_result.observed - stl_seasonal
            
            ## MK trend
            criti_slp = _ADF_MK.MK(df_criti_ave_s, 0.1)
            # criti_slp = _ADF_MK.MK(stl_deseason, 0.1)
            
            ## ADF and slope
            criti_slp_adf = _ADF_MK.ADF(df_criti_ave_s) #use deseasonal data

            
            criti_result[reginame] = [criti_var, criti_month, criti_sd, criti_slp, criti_slp_adf]
            
            # ### preparation for plot
            # df_criti_ave.name = criti_var
            # df_criti_std.name = "std"
            # df_plot = pd.concat([df_criti_ave,df_criti_std], axis=1)
            
            
            """ #Plot time series all vars"""
            """ again prepare dataset """
            fig,axes = plt.subplots(len(varlist),1, figsize=(10, 10))
            fig.subplots_adjust(hspace=0.5)  
            for i,var2 in enumerate(varlist):
                
                df_var2_list = []
                for csvfi in csv_list:
                    df_csv = _csv_to_dataframe.main(csvfi)
                    # df_csv_var_dic = {}
                    df_var = df_csv[var2]
                    df_var_month = df_var[df_var.index.month==criti_month]
                    df_var_month = df_var_month[~np.isnan(df_var_month)]
                    df_var2_list.append(df_var_month)
                
                df_var_conc = pd.concat(df_var2_list,axis=1)
                df_var_mean = df_var_conc.mean(axis=1)
                df_var_std = df_var_conc.std(axis=1)
                df_var_mean.name = var2
                df_var_std.name = "std"
                
                """ plot in ax"""
                ax = axes[i]
                ax.errorbar(df_var_mean.index, df_var_mean.values, yerr=df_var_std.values, 
                            label = f"{var2}",color='dodgerblue', fmt='-o', capsize=3, ecolor='lightgrey')
                ax.tick_params(axis='y', labelsize=14)
                ax.set_ylabel(f"{var2}", fontsize = 14)
                if i == len(varlist)-1:
                    ax.tick_params(axis='x', labelsize=10)
                    plt.xticks(df_var_mean.index, rotation=45)
                else:
                    ax.tick_params(axis='x', labelsize=0)
                if i ==0:
                    ax.legend(fontsize=10, loc='lower center', bbox_to_anchor=(.85, 1.0),frameon=False)
                    ax.set_title(f"{reginame} {criti_var} {month_calendar[criti_month]}")
            plt.tight_layout()
            ### Export fig
            out_dir_fig = out_dir + os.sep + "_png"
            os.makedirs(out_dir_fig, exist_ok=True)
            fig.savefig(out_dir_fig + os.sep + f"{reginame_csv}.png")
            plt.close()
                
            # fig = plt.figure(figsize=(12, 5))
            # fig.subplots_adjust()
            # ax = fig.add_subplot(1,1,1)
            # # ax.plot(df_criti_ave.index, df_criti_ave) #color='black'
            # ax.errorbar(df_plot.index, df_plot[criti_var].values, yerr=df_plot[f"std"].values, 
            #             color='black', fmt='-o', capsize=3) #label = f"average during {ei}"
            # ax.tick_params(axis='x', labelsize=14)
            # ax.tick_params(axis='y', labelsize=14)
            # # plt.xticks(df_var.index, rotation=45)
            # ax.set_ylabel(f"{criti_var} {units[criti_var]}", fontsize = 14)
            # plt.legend(fontsize=14)
            # plt.tight_layout()
            # ### Export fig
            # out_dir_fig = out_dir + os.sep + "_png"
            # os.makedirs(out_dir_fig, exist_ok=True)
            # fig.savefig(out_dir_fig + os.sep + f"critical_{reginame_csv}_{criti_var}_{criti_month}.png")
            # plt.close()
        
### Export results of all region
df_result = pd.DataFrame.from_dict(criti_result).T
df_result= df_result.rename(columns={0:"var",1:"month",2:"std",3:"MKslope",4:"ADFslope"})
df_result.to_csv(out_dir + os.sep + "std_slope.csv")
        
        
## ------------------------------------------                
""" MKslope and CV plot by region"""
## ------------------------------------------
anomaly_csv = out_dir + os.sep + "std_slope.csv"
# anomaly_csv = r"D:\Malaysia\02_Timeseries\YieldWater\05_var_variation_ENSO\_partial\anomalies_ENSO_IOD.csv"

varlist = ['rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD'] #'GOSIF', 
## set yield dictionary for color
gdf_region = gpd.read_file(shp_region)
gdf_region = gdf_region.drop(columns=['NAME_1','ADM1_EN','rank_yield']) #'slope',

df_anomaly = pd.read_csv(anomaly_csv, index_col=0)
df_yield_slope = gdf_region.set_index("Name").slope #looks same Name order
dic_yield_slope = df_yield_slope.to_dict()

""" Plot"""
def collect_ano(tarvar):
    both_ano = {}
    cv_var = {}
    slope_var = {}
    slopeadf_var = {}
    for regi, row in df_anomaly.iterrows():
        var = row["var"]
        if var == tarvar:
            cv_ano = row["std"]
            slope_ano = row["MKslope"]
            slopeadf_ano = row["ADFslope"]
            cv_var[regi]=cv_ano
            slope_var[regi]=slope_ano
            slopeadf_var[regi] =slopeadf_ano
            
    cv_var_sorted = dict(sorted(cv_var.items(), key=lambda item: item[1]))
    slope_var_sorted = dict(sorted(slope_var.items(), key=lambda item: item[1]))
    slopeadf_var_sorted = dict(sorted(slopeadf_var.items(), key=lambda item: item[1]))
    both_ano[tarvar] = [cv_var_sorted, slope_var_sorted, slopeadf_var_sorted]
    
    return both_ano
         

cv_ano_all = {}
slope_ano_all = {}
slopeadf_ano_all = {}
for var in varlist:
    ano_dic = collect_ano(var) #enso, iod
    cv_ano_all[var] = ano_dic[var][0]
    slope_ano_all[var] = ano_dic[var][1]
    slopeadf_ano_all[var] = ano_dic[var][1]

var_to_ax_idx = {
    "rain": (0, 0),
    "temp": (0, 1),
    "VPD": (1, 0),
    "Et": (1, 1),
    "Eb": (2, 0),
    "SM": (2, 1),
    "VOD": (3, 0)}

var_name = {
    "rain": "Precipitation",
    "temp": "Temperature",
    "VPD": "VPD",
    "Et": "Transpiration",
    "Eb": "Evaporation",
    "SM": "Soil moisture",
    "VOD": "VOD"}

def get_minmax(tardic):
    #tardic = cv_ano_all
    dicvalues = list(tardic.values())
    allvalues = [list(di.values()) for di in dicvalues]
    allvalues_flat = []
    for li in allvalues:
        for li2 in li:
            allvalues_flat.append(li2)
    minlistval = min(allvalues_flat)
    maxlistval = max(allvalues_flat)
    return minlistval, maxlistval
    

region_short = {
    "Johor":"Johor", "Kedah":"Kedah", "Kelantan":"Kelantan","Melaka":"Melaka",
    "Negeri Sembilan":"Neger S","Pahang":"Pahang","Perak":"Perak","Pulau Pinang":"Pulau P",
    "Sabah":"Sabah", "Sarawak":"Sarawak","Selangor":"Selangor", "Terengganu":"Terengganu",
    "Aceh":"Aceh", "Banten":"Banten", "Bengkulu":"Bengkulu", "Gorontalo":"Gorontalo",
    "Jambi":"Jambi", "Jawa Barat":"Jawa B", 
    "Kalimantan Barat":"Kalim B","Kalimantan Selatan":"Kalim S","Kalimantan Tengah":"Kalim Te", 
    "Kalimantan Timur":"Kalim Ti", "Kalimantan Utara":"Kalim U",
    "Kepulauan Bangka Belitung":"Kepul BB", "Kepulauan Riau":"Kepul R",
    "Lampung":"Lampung", "Maluku":"Maluku", "Papua":"Papua", "Papua Barat":"Papua B",
    "Riau":"Riau", "Sulawesi Barat":"Sulaw B", "Sulawesi Selatan":"Sulaw S",
    "Sulawesi Tengah":"Sulaw Th", "Sulawesi Tenggara":"Sulaw Tr", 
    "Sumatera Barat":"Sumat B", "Sumatera Selatan":"Sumat S", "Sumatera Utara":"Sumat U"
    }


## 全部プロットver (region名前を省略する)
def plot_anomaly(anodic, filename, minr, maxr):
    fig,axes = plt.subplots(4,2, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.5)
    ## yield for color
    cmap_bar = plt.get_cmap('coolwarm')
    # extracted_yield = {key: dic_yield_slope[key] for key in regi_list if key in dic_yield_slope}
    yield_list = list(dic_yield_slope.values())
    norm=colors.TwoSlopeNorm(vmin=min(yield_list), vcenter=0., vmax=max(yield_list))
    # RGB情報に変換
    color_dic = {}
    for reg, slp in dic_yield_slope.items():
        color_dic[reg] = cmap_bar(norm(slp))
    ##
    for var in varlist:
        anodic2 = anodic[var]
        regi_list = list(anodic2.keys())
        regi_short = [region_short[r] for r in regi_list] #replace to short name
        ano_list = list(anodic2.values())
        ## バーの太さを揃えるため最低3要素入れる（空でも）
        if len(regi_short)<3:
            regi_short = regi_short + ["",""]
            ano_list = ano_list + [np.nan,np.nan]
        extracted_color = {key: color_dic[key] for key in regi_list if key in color_dic}
        row, col = var_to_ax_idx[var]
        ax = axes[row, col]
        num_bars = len(regi_short) #max 10
        bar_width = 0.5 * (num_bars/10)
        ax.bar(regi_short, ano_list, width=bar_width, color=extracted_color.values())
        ax.axhline(y=0, color='grey', linewidth=0.5) #linestyle='--',
        ax.set_xticks(range(len(regi_short)))
        ax.set_xticklabels(regi_short, rotation=30, fontsize=14)
        ax.set_title(var_name[var], fontsize=16)
        ax.set_ylim(minr,maxr)
    fig.delaxes(axes[3, 1])
    ## add yield color bar
    cbar_ax = fig.add_axes([0.6, 0.15, 0.3, 0.02]) #left, bottom, width, height
    sm = plt.cm.ScalarMappable(cmap=cmap_bar, norm=norm)
    sm.set_array([])  # Dummy array for ScalarMappable
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Yield slope', y=1, labelpad=10, fontsize=16) #タイトル上にならない
    plt.tight_layout()
    ### Export fig
    fig.savefig(out_dir + os.sep + f"{filename}.png")
    plt.close()
    
## Run----------
minrange,maxrange = get_minmax(cv_ano_all)
plot_anomaly(cv_ano_all, "CV", minrange,0.3)

minrange,maxrange = get_minmax(slope_ano_all)
plot_anomaly(slope_ano_all, "MKslope", minrange, 0.05)    

minrange,maxrange = get_minmax(slopeadf_ano_all)
plot_anomaly(slopeadf_ano_all, "ADFslope", minrange, 0.05) 



## ひとつずつプロットver　## 体裁まとめてプロットに合わせて直してない
def plot_anomaly(anodic, filename, minr, maxr):
    ## yield for color
    cmap_bar = plt.get_cmap('coolwarm')
    # extracted_yield = {key: dic_yield_slope[key] for key in regi_list if key in dic_yield_slope}
    yield_list = list(dic_yield_slope.values())
    norm=colors.TwoSlopeNorm(vmin=min(yield_list), vcenter=0., vmax=max(yield_list))
    # RGB情報に変換
    color_dic = {}
    for reg, slp in dic_yield_slope.items():
        color_dic[reg] = cmap_bar(norm(slp))
        
    for var in varlist:
        fig,axes = plt.subplots(figsize=(20,10))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.rcParams['figure.subplot.bottom'] = 0.5
        anodic2 = anodic[var]
        regi_list = list(anodic2.keys())
        ano_list = list(anodic2.values())
        extracted_color = {key: color_dic[key] for key in regi_list if key in color_dic}
        ax = axes
        num_bars = len(regi_list) #max 10
        bar_width = 0.5 * (num_bars/10)
        ax.bar(regi_list, ano_list, width=bar_width, color=extracted_color.values())
        ax.axhline(y=0, color='grey', linewidth=0.5) #linestyle='--',
        ax.set_xticks(range(len(regi_list)))
        ax.set_xticklabels(regi_list, rotation=90, fontsize=30)
        # ax.set_title(var, fontsize=20)
        ax.set_ylim(minr,maxr)
        ax.tick_params(axis='y', labelsize=16)
        ## add yield color bar
        cbar_ax = fig.add_axes([0.88, 0.4, 0.02, 0.55]) #left, bottom, width, height #0.6, 0.15, 0.3, 0.02
        # cbar_ax = fig.add_axes([0.2, 0.15, 0.6, 0.03])
        sm = plt.cm.ScalarMappable(cmap=cmap_bar, norm=norm)
        sm.set_array([])  # Dummy array for ScalarMappable
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation="vertical")
        cbar.ax.tick_params(labelsize=16) 
        cbar.set_label('Yield slope', labelpad=14, fontsize=30, loc='center')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) #left,bottom,right,top
        ### Export fig
        fig.savefig(out_dir + os.sep + f"{filename}_{var}.png")
        plt.close()


## Run----------
minrange,maxrange = get_minmax(cv_ano_all)
plot_anomaly(cv_ano_all, "CV", minrange,maxrange)

minrange,maxrange = get_minmax(slope_ano_all)
plot_anomaly(slope_ano_all, "MKslope", minrange, maxrange)    

minrange,maxrange = get_minmax(slopeadf_ano_all)
plot_anomaly(slopeadf_ano_all, "ADFslope", minrange, maxrange)  

    
            
        
        


