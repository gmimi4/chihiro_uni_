# -*- coding: utf-8 -*-
"""
Composite plot all vars through months during ENSO and IOD
"""
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import glob
import matplotlib.pyplot as plt
import calendar
import itertools
# os.chdir(r"C:\Users\chihiro\Desktop\Python\ANALYSIS\YieldWater")
os.chdir("/Users/wtakeuchi/Desktop/Python/ANALYSIS/YieldWater")
import _yield_csv  
import _csv_to_dataframe 
import _find_csvs

# enso_csv = r"F:\MAlaysia\ENSO\00_download\meiv2.csv"
# iod_csv = r"F:\MAlaysia\ENSO\10_IOD\NASA_Json.csv"
# pearson_dir = r"D:\Malaysia\02_Timeseries\YieldWater\01_correlation_timelag\_partial"
# shp_region = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
# shp_extent = r"F:\MAlaysia\AOI\extent\Malaysia_and_Indonesia_extent_divided.shp"
# shp_01grid_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index"
# shp_grid = shp_01grid_dir + os.sep + "grid_01degree_210_496.shp"
# palm_txt2002 = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index\grid_01degree_210_496_palm2002.txt"
# var_csv_dir = r"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels_until2023"
# out_dir = r"D:\Malaysia\02_Timeseries\YieldWater\05_var_variation_ENSO\_partial\_composite_plot"

enso_csv = '/Volumes/PortableSSD/Malaysia/ENSO/00_download/meiv2.csv'
iod_csv = '/Volumes/PortableSSD/Malaysia/ENSO/10_IOD/NASA_Json.csv'
pearson_dir = "/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/01_correlation_timelag/_partial" 
shp_region = "/Volumes/SSD_2/Malaysia/Validation/1_Yield_doc/shp/region_slope_fin.shp"
shp_extent = "/Volumes/PortableSSD/Malaysia/AOI/extent/Malaysia_and_Indonesia_extent_divided.shp"
shp_01grid_dir = "/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/0_palm_index_EVI"
shp_grid = shp_01grid_dir + os.sep + "grid_01degree_210_491.shp"
palm_txt2002 = "/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/0_palm_index_EVI/grid_01degree_210_491_palm2002.txt"
var_csv_dir = "/Volumes/PortableSSD/Malaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels_EVI"
out_dir = '/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/05_var_variation_ENSO/_composite_plot_years'
# shp_01grid_dir = "/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/0_palm_index"
# shp_grid = shp_01grid_dir + os.sep + "grid_01degree_210_496.shp"
# palm_txt2002 = "/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/0_palm_index/grid_01degree_210_496_palm2002.txt"
# var_csv_dir = "/Volumes/PortableSSD/Malaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels_until2023"
# out_dir = '/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/05_var_variation_ENSO/_composite_plot_years'

varlist = ['rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD'] #'GOSIF',
units = {'':"",'GOSIF':"W/m2/μm/sr/month", 'rain':"mm", 'temp':"degreeC", 
          'VPD':"hPa", 'Et':"mm/day", 'Eb':"mm/day", 'SM':"m3/m3", 'VOD':""}
month_calendar = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec" }


startdate = datetime(2002,1,1)
enddate = datetime(2023,12,31)

### Extract date when exceeding threhold
def threshold_to_date(seri,thre):## threshold
    df_valid = seri.where(seri>thre) #|(seri<thre*-1)
    df_valid = df_valid.dropna()
    valid_date = list(df_valid.index)
    # convert date to end of month
    valid_date = [ts + pd.offsets.MonthEnd(0) for ts in valid_date]
    valid_date = list(set(valid_date))
    return valid_date

""" # prepare ENSO timeseries """
enso_thre = 0.5
df_mei = pd.read_csv(enso_csv)
df_mei = df_mei.set_index("YEAR")

### # convert mei str to datetime 
mei_month_dict = {"DJ":1,"JF":2,"FM":3,"MA":4,"AM":5,"MJ":6,
                  "JJ":7,"JA":8,"AS":9,"SO":10,"ON":11,"ND":12}

elnino_list, lanina_list, neutral_list = [],[],[]

for year, row in df_mei.iterrows():
    for colname, value in row.items():
        if value >enso_thre:
            elnino_list.append([year, colname])
        elif value < -1 * enso_thre:
            lanina_list.append([year, colname])
        else:
            neutral_list.append([year, colname])

## filtering period
elnino_list = [e for e in elnino_list if (int(e[0])>=startdate.year)&(int(e[0])<=enddate.year)]

""" # convert mei str to datetime """
def convert_mei_date(mei_list):
    mei_list_date = []
    for me in mei_list:
        yearint = int(me[0])
        mei_mon = mei_month_dict[me[1]]
        lastday1 = calendar.monthrange(yearint, mei_mon) #num of weeks, days
        yyyymmdd_1 =  datetime(yearint, mei_mon, lastday1[1])
        mei_list_date.append([yyyymmdd_1])
    
    mei_list_date = list((itertools.chain.from_iterable(mei_list_date)))
    mei_list_date = sorted(list(set(mei_list_date)))
    
    return mei_list_date
											    
### convert to datetime ###
elnino_list_date = convert_mei_date(elnino_list)


### ElNino from to #start from 0.5< till 1 year later
elnino_years = list(set([y.year for y in elnino_list_date]))
elnino_from_to =  []
for yr in elnino_years:
    elenino_months = [e for e in elnino_list_date if e.year==yr ]
    elenino_earliest = min(elenino_months)
    elenino_end = elenino_earliest + relativedelta(months=11)
    elnino_from_to.append([elenino_earliest, elenino_end])
    
#make throughout period
elnino_datetimes_ = [pd.date_range(start=period[0], end=period[1]).to_list() for period in elnino_from_to]
elnino_datetimes = []
for sublist in elnino_datetimes_:
    for dt in sublist:
        elnino_datetimes.append(dt)

elnino_datetimes_monthly = [d + pd.offsets.MonthEnd(0) for d in elnino_datetimes]
elnino_datetimes_monthly = list(set(elnino_datetimes_monthly))
elnino_datetimes_monthly = sorted(elnino_datetimes_monthly)
elnino_datetimes_monthly = pd.DatetimeIndex(elnino_datetimes_monthly)


# """ # prepare IOD timeseries """
# df_iod = pd.read_csv(iod_csv)
# df_iod['date'] = pd.to_datetime(df_iod['datetime'])
# df_iod = df_iod.drop("datetime",axis=1)
# df_iod = df_iod.rename(columns={"date":"datetime"})
# df_iod = df_iod.set_index("datetime")
# ## period
# df_iod = df_iod.loc[startdate:enddate]
# ## threshold
# df_iod_std = abs(df_iod).std()
# df_iod_mean = abs(df_iod).mean()
# iod_thre = df_iod_mean + df_iod_std
# iod_valid_date = threshold_to_date(df_iod,iod_thre)

""" prepare polygon"""
gdf_region = gpd.read_file(shp_region) 
gdf_region = gdf_region.drop(columns=['NAME_1','ADM1_EN','slope','rank_yield'])
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

""" FFB yield df"""
df_yield, df_yield_z, df_yield_det = _yield_csv.main()
nondata_region = list(df_yield_z[df_yield_z.isna().all(axis=1)].index)
nondata_region = nondata_region +["Maluku Utara"]


## -----------------------------
""" Composite plot"""
## -----------------------------

""" find hihest pearson var and month"""
def get_key(my_dict, val):
    for key, value in my_dict.items():
        if value == val:
            return key

### averaging by monh through all period
def averaging_by_month(df):
    df_ave = df.groupby(df.index.month).mean()
    df_std = df.groupby(df.index.month).std()
    df_std.name = f"{var}std"
    df_fin = pd.concat([df_ave, df_std],axis=1)
    return df_fin
        
pearson_csvs = glob.glob(pearson_dir + os.sep + "*_abs.csv")

enso_result_region = {}
iod_result_region = {}
for i, row in tqdm(gdf_region.iterrows()):
    # row = gdf_region.loc[i,:]
    regipoly = row.geometry
    reginame = row.Name
    reginame_csv = reginame.replace(" ", "")
    ## pass if no data region
    if reginame in nondata_region:
        continue
    else:
        csv_pearson = [c for c in pearson_csvs if reginame_csv in c][0]
        """ find highest pearson var and mont"""
        df_peason = pd.read_csv(csv_pearson, index_col=0)
        # df_peason = df_peason.drop("GOSIF", axis=1)
        df_peason_abs = df_peason.abs()
        
        if len(df_peason_abs.dropna(how='all')) == 0:
            continue
        else:
            # ### Find critical var name *just for filename
            # criti_var = df_peason_abs.max().idxmax()
            # criti_month = df_peason_abs[criti_var].idxmax() #month str
            # criti_month = get_key(month_calendar, criti_month.split('_')[0])
            
            
            """ Extract timeseries data of critical var at ENSO period"""
            csv_list = _find_csvs.main(regipoly,gdf_region,gdf_grid, gdf_extent, gdf_A_dic, list_palm, var_csv_dir)
            
            enso_result = []
            # iod_result = []
            for var in varlist:
                df_csv_list = []
                for csvfi in csv_list:
                    df_csv = _csv_to_dataframe.main(csvfi)
                    ### collect df_csv for averaging by var            
                    df_var = df_csv[var]
                    df_csv_list.append(df_var)
                    
                
                """ averaging timeseries data"""
                df_concat = pd.concat(df_csv_list,axis=1)
                df_var_ave = df_concat.mean(axis=1)
                df_var_ave.name = var

                """ extracting enso period"""
                elnino_dates = df_var_ave.index.intersection(elnino_datetimes_monthly)
                non_dates = df_var_ave.index.difference(elnino_datetimes_monthly)
                
                df_var_enso = df_var_ave[df_var_ave.index.isin(elnino_dates)]
                # df_var_iod = df_var_ave[df_var_ave.index.isin(iod_valid_date)]
                df_var_noenso = df_var_ave[df_var_ave.index.isin(non_dates)]## non enso iod period
                # df_var_noiod = df_var_ave[~df_var_ave.index.isin(iod_valid_date)]
                
                """ averaging by month"""
                df_var_enso_fin = averaging_by_month(df_var_enso)
                # df_var_iod_fin = averaging_by_month(df_var_iod)
                df_var_nonenso_fin = averaging_by_month(df_var_noenso)
                # df_var_noniod_fin = averaging_by_month(df_var_noiod)
                
                df_var_nonenso_fin = df_var_nonenso_fin.rename(columns={var:f"non{var}",f"{var}std":f"non{var}std"})
                # df_var_noniod_fin = df_var_noniod_fin.rename(columns={var:f"non{var}",f"{var}std":f"non{var}std"})
                    
                ### Collect
                enso_result.append(pd.concat([df_var_enso_fin, df_var_nonenso_fin],axis=1))
                # iod_result.append(pd.concat([df_var_iod_fin, df_var_noniod_fin],axis=1))
                
            ### Export
            df_enso_region = pd.concat(enso_result, axis=1)
            # df_iod_region = pd.concat(iod_result, axis=1)
            df_enso_region.to_csv(out_dir + os.sep + f"{reginame}_enso_composite.csv")
            # df_iod_region.to_csv(out_dir + os.sep + f"{reginame}_{criti_var}_{criti_month}_iod_composite.csv")
            
            ## Collect
            enso_result_region[reginame] = df_enso_region
            # iod_result_region[reginame] = df_iod_region
            
            # """ Plot"""
            # x_label = list(month_calendar.values())
            # for ei, result in {"ENSO":df_enso_region, "IOD":df_iod_region}.items():
            #     fig,axes = plt.subplots(len(varlist),1, figsize=(10, 50))
            #     fig.subplots_adjust(hspace=0.5)  
            #     for i,var in enumerate(varlist):
            #         ax = axes[i]
            #         # enso or iod
            #         df_plot = result[[var,f"{var}std"]].sort_index()
            #         # non enso or non iod
            #         df_plot_non = result[[f"non{var}",f"non{var}std"]].sort_index()
    
            #         ax.errorbar(df_plot.index, df_plot[var].values, yerr=df_plot[f"{var}std"].values, 
            #                     label = f"average during {ei}",color='black', fmt='-o', capsize=3)
            #         ax.errorbar(df_plot_non.index, df_plot_non[f"non{var}"].values, yerr=df_plot_non[f"non{var}std"].values, 
            #                     label = f"average during non-{ei}", color='grey', fmt=':o', capsize=3)
            #         ax.tick_params(axis='y', labelsize=14)
            #         ax.set_ylabel(f"{var}", fontsize = 14)
            #         if i == len(varlist)-1:
            #             ax.tick_params(axis='x', labelsize=14)
            #             plt.xticks(df_plot.index, x_label, rotation=45)
            #         else:
            #             ax.tick_params(axis='x', labelsize=0)
            #         if i ==0:
            #             ax.legend(fontsize=10, loc='lower center', bbox_to_anchor=(.85, 1.0),frameon=False)
            #             ax.set_title(f"{reginame} {criti_var} {month_calendar[criti_month]}")
            #         plt.tight_layout()
            #         ### Export fig
            #         out_dir_fig = out_dir + os.sep + "_png"
            #         os.makedirs(out_dir_fig, exist_ok=True)
            #         fig.savefig(out_dir_fig + os.sep + f"{ei}_{reginame_csv}.png")
            #         plt.close()
                
                
            
""" Plot later # なぜか上記のプログラム中で出すとfigsizaがおかしい"""
# out_dir = "/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/05_var_variation_ENSO/_partial/_composite_plot"
csvs_enso = glob.glob(out_dir + os.sep + "*_enso_composite.csv")
# csvs_iod = glob.glob(out_dir + os.sep + "*_iod_composite.csv")
regions = [os.path.basename(c).split("_")[0] for c in csvs_enso]

x_label = list(month_calendar.values())

for regi in tqdm(regions):
    csvenso = [c for c in csvs_enso if regi in c][0]
    df_enso_region = pd.read_csv(csvenso, index_col=0)
    for ei, result in {"ENSO":df_enso_region}.items():
        fig,axes = plt.subplots(len(varlist),1, figsize=(10, 50))
        fig.subplots_adjust(hspace=0.5)  
        for i,var in enumerate(varlist):
            ax = axes[i]
            # enso or iod
            df_plot = result[[var,f"{var}std"]].sort_index()
            # non enso or non iod
            df_plot_non = result[[f"non{var}",f"non{var}std"]].sort_index()
            if ei =="ENSO":
                lavel_ei = "ElNino"
            ax.errorbar(df_plot.index, df_plot[var].values, yerr=df_plot[f"{var}std"].values, 
                        label = f"average during {lavel_ei}",color='black', fmt='-o', capsize=3)
            ax.errorbar(df_plot_non.index, df_plot_non[f"non{var}"].values, yerr=df_plot_non[f"non{var}std"].values, 
                        label = f"average during non-{lavel_ei}", color='grey', fmt=':o', capsize=3)
            ax.tick_params(axis='y', labelsize=14)
            ax.set_ylabel(f"{var}", fontsize = 14)
            if i == len(varlist)-1:
                ax.tick_params(axis='x', labelsize=14)
                plt.xticks(df_plot.index, x_label, rotation=45)
            else:
                ax.tick_params(axis='x', labelsize=0)
            if i ==0:
                ax.legend(fontsize=10, loc='lower center', bbox_to_anchor=(.85, 1.0),frameon=False)
                # ax.set_title(f"{regi} {criti_var} {month_calendar[criti_month]}")
            plt.tight_layout()
            ### Export fig
            out_dir_fig = out_dir + os.sep + "_png"
            os.makedirs(out_dir_fig, exist_ok=True)
            reginame = regi.replace(" ","")
            fig.savefig(out_dir_fig + os.sep + f"{ei}_{reginame}.png")
            plt.close()
        
        


