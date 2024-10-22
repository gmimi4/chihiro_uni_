# -*- coding: utf-8 -*-
"""
Plot of critical var in critical month with ENSO and IOD
"""
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import datetime
import glob
import matplotlib.pyplot as plt
from matplotlib import colorbar, colors
import math
os.chdir(r"C:\Users\chihiro\Desktop\Python\ANALYSIS\YieldWater")
import _yield_csv  
import _csv_to_dataframe 
import _find_csvs

enso_csv = r"F:\MAlaysia\ENSO\00_download\meiv2.csv"
iod_csv = r"F:\MAlaysia\ENSO\10_IOD\NASA_Json.csv"

pearson_dir = r"D:\Malaysia\02_Timeseries\YieldWater\01_correlation_timelag\_partial"
shp_region = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
shp_extent = r"F:\MAlaysia\AOI\extent\Malaysia_and_Indonesia_extent_divided.shp"
shp_01grid_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index"
shp_grid = shp_01grid_dir + os.sep + "grid_01degree_210_496.shp"
palm_txt2002 = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index\grid_01degree_210_496_palm2002.txt"
var_csv_dir = r"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels_until2023"
out_dir = r"D:\Malaysia\02_Timeseries\YieldWater\05_var_variation_ENSO\_partial"

varlist = ['rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD'] #'GOSIF', 
units = {'GOSIF':"W/m2/μm/sr/month", 'rain':"mm", 'temp':"degreeC", 
          'VPD':"hPa", 'Et':"mm/day", 'Eb':"mm/day", 'SM':"m3/m3", 'VOD':""}
month_calendar = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec" }


startdate = datetime.datetime(2002,1,1)
enddate = datetime.datetime(2023,12,31)

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
df_enso = pd.read_csv(enso_csv)
df_enso = df_enso.set_index("YEAR")

### # convert mei str to datetime 
mei_month_dict = {"DJ":1,"JF":2,"FM":3,"MA":4,"AM":5,"MJ":6,
                  "JJ":7,"JA":8,"AS":9,"SO":10,"ON":11,"ND":12}

df_enso_num = df_enso.rename(columns=mei_month_dict)

enso_result =[]
for yr,row in df_enso_num.iterrows():
    for m in row.index:
        datetime_ = datetime.datetime(int(yr), m, 1)
        mei_val= row.at[m]
        enso_result.append([datetime_, mei_val])
        
df_enso_series =pd.DataFrame(enso_result, columns=["datetime","enso"])
df_enso_series = df_enso_series.set_index("datetime")
## period
df_enso_series = df_enso_series.loc[startdate:enddate]
## threshold
enso_valid_date = threshold_to_date(df_enso_series, 0.5)


""" # prepare IOD timeseries """
df_iod = pd.read_csv(iod_csv)
df_iod['date'] = pd.to_datetime(df_iod['datetime'])
df_iod = df_iod.drop("datetime",axis=1)
df_iod = df_iod.rename(columns={"date":"datetime"})
df_iod = df_iod.set_index("datetime")
## period
df_iod = df_iod.loc[startdate:enddate]
## threshold
df_iod_std = abs(df_iod).std()
df_iod_mean = abs(df_iod).mean()
iod_thre = df_iod_mean + df_iod_std
iod_valid_date = threshold_to_date(df_iod,iod_thre)

""" prepare polygon"""
gdf_region = gpd.read_file(shp_region) 
gdf_region = gdf_region.drop(columns=['NAME_1','ADM1_EN','rank_yield'])
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
df_yield, df_yield_z = _yield_csv.main()
nondata_region = list(df_yield_z[df_yield_z.isna().all(axis=1)].index)


## -----------------------------
""" Composite plot"""
## -----------------------------

""" find hihest pearson var and month"""
def get_key(my_dict, val):
    for key, value in my_dict.items():
        if value == val:
            return key
        
pearson_csvs = glob.glob(pearson_dir + os.sep + "*_abs.csv")

criti_result = {}
for i, row in tqdm(gdf_region.iterrows()):
    regipoly = row.geometry
    reginame = row.Name
    ## pass if no data region
    if reginame in nondata_region:
        continue
    else:
        reginame_csv = reginame.replace(" ", "")
        csv_pearson = [c for c in pearson_csvs if reginame_csv in c][0]
        """ find hihest pearson var and mont"""
        df_peason = pd.read_csv(csv_pearson, index_col=0)
        ## drop GOSIF
        # df_peason = df_peason.drop("GOSIF", axis=1)
        df_peason_abs = df_peason.abs()
        
        if len(df_peason_abs.dropna()) == 0:
            criti_result[reginame] = ["", 0, np.nan, np.nan]
        else:
            ### Find critical var name and month
            criti_var = df_peason_abs.max().idxmax()
            criti_month = df_peason_abs[criti_var].idxmax() #month str
            criti_month = get_key(month_calendar, criti_month.split('_')[0])
            
            
            """ Extract timeseries data of critical var at ENSO period"""
            csv_list = _find_csvs.main(regipoly,gdf_region,gdf_grid, gdf_extent, gdf_A_dic, list_palm, var_csv_dir)
            
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
            df_criti_concat = pd.concat(df_csv_list,axis=1)
            df_criti_ave = df_criti_concat.mean(axis=1)
            df_criti_ave.name = criti_var
            df_criti_std = df_criti_concat.std(axis=1)
            df_criti_std.name = "std"
            df_criti_fin = pd.concat([df_criti_ave,df_criti_std],axis=1)
            
            # df_criti_concat_s = pd.concat(df_csv_list_s,axis=1)
            # df_criti_ave_s = df_criti_concat_s.mean(axis=1)
            
            ## mean all period in this csv for comparison
            mean_csv = df_criti_ave.mean()
            
            """ extracting enso period"""
            df_criti_ave_enso = df_criti_fin.loc[df_criti_ave.index.isin(enso_valid_date)]
            
            """ extracting IOD period"""
            df_criti_ave_iod = df_criti_fin.loc[df_criti_ave.index.isin(iod_valid_date)]

            """ Anomaly"""
            #enso
            devi_enso = df_criti_ave_enso[criti_var].mean() - mean_csv
            devi_enso_rate = devi_enso/mean_csv
            #iod
            devi_iod = df_criti_ave_iod[criti_var].mean() - mean_csv
            devi_iod_rate = devi_iod/mean_csv
            
            
            criti_result[reginame] = [criti_var, criti_month, devi_enso_rate, devi_iod_rate]
    
                
            
## Export
df_result = pd.DataFrame.from_dict(criti_result).T
df_result= df_result.rename(columns={0:"var",1:"month",2:"anomalyENSO",3:"anomalyIOD"})
df_result.to_csv(out_dir + os.sep + "anomalies_ENSO_IOD.csv")
        
        
""" Anomaly plot by region"""
anomaly_csv = out_dir + os.sep + "anomalies_ENSO_IOD.csv"
# anomaly_csv = r"D:\Malaysia\02_Timeseries\YieldWater\05_var_variation_ENSO\_partial\anomalies_ENSO_IOD.csv"
df_anomaly = pd.read_csv(anomaly_csv, index_col=0)

## set yield dictionary for color
df_anomaly = pd.read_csv(anomaly_csv, index_col=0)
df_yield_slope = gdf_region.set_index("Name").slope #looks same Name order
dic_yield_slope = df_yield_slope.to_dict()

""" Plot"""
def collect_ano(tarvar):
    both_ano = {}
    enso_var = {}
    iod_var = {}
    for regi, row in df_anomaly.iterrows():
        var = row["var"]
        if var == tarvar:
            enso_ano = row["anomalyENSO"]
            iod_ano = row["anomalyIOD"]
            enso_var[regi]=enso_ano
            iod_var[regi]=iod_ano
            
    enso_var_sorted = dict(sorted(enso_var.items(), key=lambda item: item[1]))
    iod_var_sorted = dict(sorted(iod_var.items(), key=lambda item: item[1]))
    both_ano[tarvar] = [enso_var_sorted, iod_var_sorted]
    
    return both_ano
         

enso_ano_all = {}
iod_ano_all = {}
for var in varlist:
    ano_dic = collect_ano(var) #enso, iod
    enso_ano_all[var] = ano_dic[var][0]
    iod_ano_all[var] = ano_dic[var][1]

var_to_ax_idx = {
    "rain": (0, 0),
    "temp": (0, 1),
    "VPD": (1, 0),
    "Et": (1, 1),
    "Eb": (2, 0),
    "SM": (2, 1),
    "VOD": (3, 0)}

def is_real_number(value):
    return isinstance(value, (int, float)) and not math.isnan(value)

def get_minmax(tardic):
    #tardic = cv_ano_all
    dicvalues = list(tardic.values())
    allvalues = [list(di.values()) for di in dicvalues]
    allvalues_flat = []
    for li in allvalues:
        for li2 in li:
            allvalues_flat.append(li2)
    allvalues_flat_fin = [a for a in allvalues_flat if is_real_number(a)]
    minlistval = min(allvalues_flat_fin)
    maxlistval = max(allvalues_flat_fin)
    return minlistval, maxlistval
    
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
    for var in varlist:
        anodic2 = anodic[var]
        regi_list = list(anodic2.keys())
        ano_list = list(anodic2.values())
        extracted_color = {key: color_dic[key] for key in regi_list if key in color_dic}
        row, col = var_to_ax_idx[var]
        ax = axes[row, col]            
        num_bars = len(regi_list) #max 10
        bar_width = 0.5 * (num_bars/10)
        ax.bar(regi_list, ano_list, width=bar_width, color=extracted_color.values())
        ax.axhline(y=0, color='grey', linewidth=0.5) #linestyle='--',
        ax.set_xticks(range(len(regi_list)))
        ax.set_xticklabels(regi_list, rotation=70)
        ax.set_title(var, fontsize=12)
        ax.set_ylim(minr,maxr)
    fig.delaxes(axes[3, 1])
    ## add yield color bar
    cbar_ax = fig.add_axes([0.6, 0.15, 0.3, 0.02]) #left, bottom, width, height
    sm = plt.cm.ScalarMappable(cmap=cmap_bar, norm=norm)
    sm.set_array([])  # Dummy array for ScalarMappable
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label('Yield slope')
    plt.tight_layout()
    ### Export fig
    fig.savefig(out_dir + os.sep + f"{filename}.png")
    plt.close()


## Run----------
minrange,maxrange = get_minmax(enso_ano_all)
plot_anomaly(enso_ano_all, "ENSO_anomaly.png", minrange,maxrange)

minrange,maxrange = get_minmax(iod_ano_all)
plot_anomaly(iod_ano_all, "IOD_anomaly.png", minrange,maxrange)
