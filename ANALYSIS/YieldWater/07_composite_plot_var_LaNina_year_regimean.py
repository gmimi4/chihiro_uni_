# -*- coding: utf-8 -*-
"""
Composite plot all vars through year from start of ElNino
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
from scipy.stats import ttest_ind
os.chdir(r"C:\Users\chihiro\Desktop\Python\ANALYSIS\YieldWater")
# os.chdir("/Users/wtakeuchi/Desktop/Python/ANALYSIS/YieldWater")
import _yield_csv  
import _csv_to_dataframe 
import _find_csvs
plt.rcParams['font.family'] = "DejaVu Sans"#'Times New Roman' #"DejaVu Sans" #default


enso_csv = r"F:\MAlaysia\ENSO\00_download\meiv2.csv"
shp_region = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
out_dir = r"D:\Malaysia\02_Timeseries\YieldWater\05_var_variation_ENSO\_composite_plot_years_regimean_LaNina"
mean_dir = r"D:\Malaysia\02_Timeseries\CCM\01_region_mean\EVI"

# enso_csv = '/Volumes/PortableSSD/Malaysia/ENSO/00_download/meiv2.csv'
# iod_csv = '/Volumes/PortableSSD/Malaysia/ENSO/10_IOD/NASA_Json.csv'
# pearson_dir = "/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/01_correlation_timelag/_pearson" 
# shp_region = "/Volumes/SSD_2/Malaysia/Validation/1_Yield_doc/shp/region_slope_fin.shp"
# shp_extent = "/Volumes/PortableSSD/Malaysia/AOI/extent/Malaysia_and_Indonesia_extent_divided.shp"
# shp_01grid_dir = "/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/0_palm_index"
# shp_grid = shp_01grid_dir + os.sep + "grid_01degree_210_496.shp"
# palm_txt2002 = "/Volumes/SSD_2/Malaysia/02_Timeseries/Sensitivity/0_palm_index/grid_01degree_210_496_palm2002.txt"
# var_csv_dir = "/Volumes/PortableSSD/Malaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels_until2023"
# out_dir = '/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/05_var_variation_ENSO/_pearson/_composite_plot'

varlist = ['rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD'] #'GOSIF',
units = {'GOSIF':"W/m2/μm/sr/month", 'EVI':'','rain':"mm", 'temp':"degreeC", 
          'VPD':"hPa", 'Et':"mm/day", 'Eb':"mm/day", 'SM':"m3/m3", 'VOD':""}
month_calendar = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec" }

mei_month_dict = {"DJ":1,"JF":2,"FM":3,"MA":4,"AM":5,"MJ":6,
                  "JJ":7,"JA":8,"AS":9,"SO":10,"ON":11,"ND":12}

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


""" # Identify months of ElNino, LaNiNa, Neutral"""
enso_thre = 0.5
df_mei = pd.read_csv(enso_csv)
df_mei = df_mei.set_index("YEAR")

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
# elnino_list = [e for e in elnino_list if (int(e[0])>=startdate.year)&(int(e[0])<=enddate.year)]
elnino_list = [e for e in lanina_list if (int(e[0])>=startdate.year)&(int(e[0])<=enddate.year)]

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


""" prepare polygon"""
gdf_region = gpd.read_file(shp_region) 
gdf_region = gdf_region.drop(columns=['NAME_1','ADM1_EN','slope','rank_yield'])

""" FFB yield df"""
df_yield, df_yield_z,df_yield_detr = _yield_csv.main()
nondata_region = list(df_yield_z[df_yield_z.isna().all(axis=1)].index)
nondata_region = nondata_region +["Maluku Utara"]


## -----------------------------
""" Composite plot"""
## -----------------------------
def get_key(my_dict, val):
    for key, value in my_dict.items():
        if value == val:
            return key

""" # averaging by monh through all period """
def averaging_by_month(df):
    df_ave = df.groupby(df.index.month).mean()
    df_std = df.groupby(df.index.month).std()
    df_std = df_std.rename(columns ={f"{var}el":f"{var}std",f"{var}non":f"{var}nonstd"})
    df_fin = pd.concat([df_ave, df_std],axis=1)
    return df_fin


""" compare el and nonel"""
def compare_anomaly(df,var):
    sig_mons = {}
    for mon in month_calendar.keys():
        df_mon = df[df.index.month==mon]
        df_mon_el = df_mon[f"{var}el"]
        df_mon_non = df_mon[f"{var}non"]
        df_mon_el = df_mon_el.dropna()
        df_mon_non = df_mon_non.dropna()
        ### t-test
        t_stat, p_value = ttest_ind(df_mon_el, df_mon_non, equal_var=False)
        if p_value<0.1:
            sig_mon = mon
        else:
            continue
        # sig_mons.append(sig_mon)
        sig_mons[sig_mon]= p_value
    
    return sig_mons
    

        
# pearson_csvs = glob.glob(pearson_dir + os.sep + "*_abs.csv")
csvs = glob.glob(mean_dir + os.sep + "*.csv")

enso_result_region = {}
for i, row in tqdm(gdf_region.iterrows()):
    # row = gdf_region.loc[i,:]
    regipoly = row.geometry
    reginame = row.Name
    ## pass if no data region
    if reginame in nondata_region:
        continue
    else:
        reginame_csv = reginame.replace(" ", "")
        # csv_pearson = [c for c in pearson_csvs if f"peason_{reginame_csv}_abs.csv" in c][0]
        csv_pearson = [c for c in csvs if f"{reginame_csv}_meanval" in os.path.basename(c)][0]
        """ find highest pearson var and mont"""
        df_csv = pd.read_csv(csv_pearson, index_col=0,parse_dates=True)
        
        """ # set period"""
        df_csv = df_csv[((df_csv.index.year >= startdate.year) & (df_csv.index.year <= enddate.year)) ]

        """ # calculate monthyl mean and anomly in whole period"""
        df_csv_use = df_csv.copy()
        vars_list = df_csv_use.columns.tolist()
        
        # monthly_mean_std_dic = {}
        for var in vars_list:
            df_var = df_csv_use.loc[:,var]
            df_csv_use[f"{var}el"] = np.nan
            for m in range(1,13):
                specific_month_rows = df_var[df_var.index.month == m]
                elnino_dates = specific_month_rows.index.intersection(elnino_datetimes_monthly)
                non_dates = specific_month_rows.index.difference(elnino_datetimes_monthly)
                specific_month_rows_el = specific_month_rows.loc[elnino_dates] #elnino
                specific_month_rows_non = specific_month_rows.loc[non_dates] #non elnino
                monthly_mean = specific_month_rows_non.mean(skipna=True)
                df_csv_use.loc[specific_month_rows.index, f"{var}el"] = specific_month_rows_el
                df_csv_use.loc[specific_month_rows.index, f"{var}non"] = specific_month_rows_non
                
        
        enso_result = []
        anomaly_result = []
        pval_result = {}
        for var in varlist:
            df_var = df_csv_use.loc[:,[f"{var}el",f"{var}non"]]
            
            """ averaging timeseries data"""
            df_var_ave = df_var.resample("M").mean() #if monthly data, no change

            # """ extracting enso period"""
            # df_enso_list = []
            # for s_e in elnino_from_to:
            #     start = s_e[0]
            #     end = s_e[1]
            #     df_enso = df_var[(df_var.index>=start)&(df_var.index<=end)]
            #     df_enso_list.append(df_enso)
                
            # df_var_enso = pd.concat(df_enso_list)
            # df_var_enso = df_var_enso.sort_index()
                        
            """ averaging by month"""
            df_var_enso_monthly = averaging_by_month(df_var_ave)
            df_var_enso_fin = df_var_enso_monthly.loc[:,[f"{var}el",f"{var}std"]]
            df_var_nonenso_fin = df_var_enso_monthly.loc[:,[f"{var}non",f"{var}nonstd"]]
            ## rename for non
            df_var_enso_fin = df_var_enso_fin.rename(columns={f"{var}el":f"{var}"})
            df_var_nonenso_fin = df_var_nonenso_fin.rename(columns={var:f"non{var}",f"{var}std":f"non{var}std"})
                
            ### Collect
            df_elnon_concat = pd.concat([df_var_enso_fin, df_var_nonenso_fin],axis=1)
            enso_result.append(df_elnon_concat)
            
            """ find significant anomaly month"""
            sig_mons_var = compare_anomaly(df_var_ave, var)
            df_elnon_concat_ano = df_elnon_concat.copy()
            df_elnon_concat_ano[f"{var}ano"] = df_elnon_concat_ano[f"{var}"] - df_elnon_concat_ano[f"{var}non"]
            df_elnon_concat_ano[f"{var}anovalid"] = df_elnon_concat_ano[f"{var}ano"].where(df_elnon_concat_ano.index.isin(sig_mons_var.keys()), np.nan)
            df_elnon_concat_ano_fin = df_elnon_concat_ano[f"{var}anovalid"]
            anomaly_result.append(df_elnon_concat_ano_fin)
            pval_result[var] = sig_mons_var
            
        
        
        ### Export data of enso and non
        df_enso_region = pd.concat(enso_result, axis=1)
        df_enso_region.to_csv(out_dir + os.sep + f"{reginame}_enso_composite.csv")
        
        ### Export pval
        df_pval = pd.DataFrame(index=range(1,13), columns=["rain","temp","VPD","Et","Eb","SM","VOD"])
        for vr, pdic in pval_result.items():
            for m,p in pdic.items():
                df_pval.loc[m, vr] = p
        df_pval.to_csv(out_dir + os.sep + f"{reginame}_pval.csv")
        
        ### Export valid anomaly
        df_anomaly_region = pd.concat(anomaly_result, axis=1)
        out_ano_dir = out_dir + os.sep + "_sig_anomaly"
        df_anomaly_region.to_csv(out_ano_dir + os.sep + f"{reginame}_ensoanomaly.csv")
        
        ## Collect
        enso_result_region[reginame] = df_enso_region
        
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
regions = [os.path.basename(c).split("_")[0] for c in csvs_enso]

csvs_pval = glob.glob(out_dir + os.sep + "*_pval.csv")
newvarlist = {'rain':"Rain", 'temp':"Temp", 'VPD':"VPD", 'Et':"Trans", 'Eb':"Evapo", 'SM':"SM", 'VOD':"VOD"}
units={'rain':"mm", 'temp':"°C", 'VPD':"kPa", 'Et':"mm", 'Eb':"mm", 'SM':"$m^3/m^3$", 'VOD':"-"}
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']

x_label = list(month_calendar.values())

for regi in tqdm(regions):
    csvenso = [c for c in csvs_enso if regi in c][0]
    df_enso_region = pd.read_csv(csvenso, index_col=0)
    csvpval = [c for c in csvs_pval if regi in c][0]
    df_pval_region = pd.read_csv(csvpval, index_col=0)
    
    for ei, result in {"ENSO":df_enso_region}.items():
        fig,axes = plt.subplots(len(varlist),1, figsize=(10, 52))
        fig.subplots_adjust(hspace=0.5)  
        for i,var in enumerate(varlist):
            ax = axes[i]
            # enso or iod
            df_plot = result[[var,f"{var}std"]].sort_index()
            # non enso or non iod
            df_plot_non = result[[f"{var}non",f"{var}nonstd"]].sort_index()
            if ei =="ENSO":
                lavel_ei = "La Nina"
            ax.errorbar(df_plot.index, df_plot[var].values, yerr=df_plot[f"{var}std"].values, 
                        label = f"average during {lavel_ei}",color='black', fmt='-o', capsize=3)
            ax.errorbar(df_plot_non.index, df_plot_non[f"{var}non"].values, yerr=df_plot_non[f"{var}nonstd"].values, 
                        label = f"average during non-{lavel_ei}", color='grey', fmt=':o', capsize=3)
            ax.tick_params(axis='y', labelsize=14)
            newvar = newvarlist[var]
            unitstr = units[var]
            ax.set_ylabel(f"{newvar}[{unitstr}]", fontsize = 14)
            if i == len(varlist)-1:
                ax.tick_params(axis='x', labelsize=14)
                plt.xticks(df_plot.index, x_label, rotation=45)
            else:
                ax.tick_params(axis='x', labelsize=0)
            if i ==0:
                ax.legend(fontsize=16, loc='lower center', bbox_to_anchor=(.5, 1.0),frameon=False,
                          ncol=2)
            ## Add significance asterisks
            pvals = df_pval_region[var] #series
            for j, p in enumerate(pvals.values):
                j,p
                if p < 0.05:
                    ax.text(j+1, df_plot[var].iloc[j] + df_plot[f"{var}std"].iloc[j] + 0.02, '**',
                            ha='center', va='bottom', fontsize=14, color='blue')
                elif p < 0.1:
                    ax.text(j + 1, df_plot[var].iloc[j] + df_plot[f"{var}std"].iloc[j] + 0.02, '*',
                            ha='center', va='bottom', fontsize=14, color='blue')
            ## If add a,b,c,..
            # ax.text(0.01, -0.25, subplot_labels[i], transform=ax.transAxes,
            #         fontsize=18, fontweight='bold', va='center', ha='left')
            plt.tight_layout()
            ### Export fig
            out_dir_fig = out_dir + os.sep + "_png"
            os.makedirs(out_dir_fig, exist_ok=True)
            reginame = regi.replace(" ","")
            fig.savefig(out_dir_fig + os.sep + f"{ei}_{reginame}.png")
            plt.close()
        
        


