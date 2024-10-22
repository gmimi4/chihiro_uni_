# -*- coding: utf-8 -*-
"""
Correlation with annual yield and ENSO index
"""
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import scipy.stats
from scipy.stats import zscore
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import glob
import datetime
from statistics import mean
import pingouin as pg
os.chdir(r"C:\Users\chihiro\Desktop\Python\ANALYSIS\YieldWater")
import _csv_to_dataframe
import _yield_csv  


shp_region = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
shp_extent = r"F:\MAlaysia\AOI\extent\Malaysia_and_Indonesia_extent_divided.shp"
shp_01grid_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index"
shp_grid = shp_01grid_dir + os.sep + "grid_01degree_210_496.shp"
palm_txt2002 = r"D:\Malaysia\02_Timeseries\Sensitivity\0_palm_index\grid_01degree_210_496_palm2002.txt"
var_csv_dir = r"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels_until2023"
enso_csv = r"F:\MAlaysia\ENSO\00_download\meiv2.csv"
iod_csv = r"F:\MAlaysia\ENSO\10_IOD\NASA_Json.csv" 
out_dir = r"D:\Malaysia\02_Timeseries\YieldWater\06_correlation_timelag_ENSO\_partial"

startdate = datetime.datetime(2002,1,1)
enddate = datetime.datetime(2023,12,1)

""" # prepare ENSO timeseries """
# enso_thre = 0.5
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
        
df_enso_series =pd.DataFrame(enso_result, columns=["datetime","ENSO"])
df_enso_series = df_enso_series.set_index("datetime")
df_enso_series = df_enso_series.loc[startdate:enddate] ## period
## change to last day of the month
df_enso_series.index = df_enso_series.index + pd.offsets.MonthEnd(0)

""" # prepare IOD timeseries """
df_iod = pd.read_csv(iod_csv)
df_iod['date'] = pd.to_datetime(df_iod['datetime'])
df_iod = df_iod.drop("datetime",axis=1)
df_iod = df_iod.rename(columns={"date":"datetime"})
df_iod = df_iod.set_index("datetime")
df_iod = df_iod.loc[startdate:enddate] ## period
## monthly
df_iod_month = df_iod.resample("M").mean()


""" # dataset of ENSO and IOD """
df_ensoiod = pd.concat([df_enso_series, df_iod_month],axis=1)
### to z score
for var in list(df_ensoiod.columns):
    df_ensoiod[f"{var}z"] = (df_ensoiod[var] - df_ensoiod[var].mean()) / df_ensoiod[var].std()


""" yield df"""
df_yield, df_yield_z = _yield_csv.main()
nondata_region = list(df_yield_z[df_yield_z.isna().all(axis=1)].index)


""" set year list"""
year_list = [y for y in range(2002,2024,1)]


""" set gdf """
gdf_region = gpd.read_file(shp_region)

        
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
                df_csv_m = df_csv_[df_csv_.index.month == int(mon)]
                
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
                    try:
                        var_yrt = df_csv_m_yrt[f"{var}z"].values[0] # This is X at year t, month m
                    except:
                        var_yrt = np.nan
                    try:
                        var_yrt_1 = df_csv_m_yrt_1[f"{var}z"].values[0] # This is X at year t-1, month m
                    except:
                        var_yrt_1 = np.nan
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
   
# varlist = ['rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD'] #'GOSIF', 
varlist = ["ENSO","IOD"]
month_calendar = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                  7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec" }

for i, row in tqdm(gdf_region.iterrows()):
    # i=31
    # row = gdf_region.loc[i]
    regipoly = row.geometry
    reginame = row.Name
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
            
            """ pearson calc """               
            pearson_var = calc_peason(df_ensoiod) ## abs
            pearson_var_list = list(pearson_var.values())
                               
            ### concat and Export csv
            df_peason_region = pd.concat(pearson_var_list,axis=1)
            df_peason_region.to_csv(out_dir + os.sep + f"partial_{regioname_fin}_abs.csv")
    
        
    """ # Plot"""
    units = {'GOSIF':"W/m2/μm/sr/month", 'rain':"mm", 'temp':"degreeC", 
              'VPD':"hPa", 'Et':"mm/day", 'Eb':"mm/day", 'SM':"m3/m3", 'VOD':""}
    
    
    fig,axes = plt.subplots(len(varlist),1, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.5)  
    for i,var in enumerate(varlist):
        ax = axes[i]
        ax.errorbar(df_peason_region.index, df_peason_region[var].values, 
                    yerr=np.array([0 for z in range(len(df_peason_region))]), color='blue', ecolor="lightgrey",
                    fmt='-o', capsize=1) #label = f"{var}", 
        ax.tick_params(axis='y', labelsize=10)
        ax.set_ylabel(f"{var} index", fontsize = 12)
        ax.legend(fontsize=14, frameon=False, loc = "upper left") #bbox_to_anchor=(.8, 0.8)
        ax.set_ylim(-1,1) #(0,1)
        ax.tick_params(axis='x', labelsize=10)
        ax.axhline(y=0, color='grey', linewidth=0.7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        if i ==0:
            ax.set_title(f"{reginame} partial correlaton")
    plt.tight_layout()
    ### Export fig
    out_dir_fig = out_dir + os.sep + "_png"
    os.makedirs(out_dir_fig, exist_ok=True)
    fig.savefig(out_dir_fig + os.sep + f"{regioname_fin}_partial.png")
    plt.close()
        
        
# """ # ミスってplot from csv"""
# csv_dir = r"D:\Malaysia\02_Timeseries\YieldWater\01_correlation_timelag\_partial"
# csvs = glob.glob(csv_dir + os.sep + "*_abs.csv")
# csvs_std = glob.glob(csv_dir + os.sep + "*_stdabs.csv")

# varlist = ['rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD'] #'GOSIF', 

# for csvf in csvs:
#     reginame = os.path.basename(csvf)[:-4].split("_")[1]
#     std_file = [f for f in csvs_std if reginame in f][0]
#     df_mean = pd.read_csv(csvf, index_col=0)
#     df_std = pd.read_csv(std_file, index_col=0)
    
#     fig,axes = plt.subplots(4,2, figsize=(20, 10))
#     fig.subplots_adjust(hspace=0.5)  
#     for i,var in enumerate(varlist):
#         row, col = divmod(i, 2)
#         ax = axes[row, col]
#         # ax = axes[i]
#         ax.errorbar(df_mean.index, df_mean[var].values, 
#                     yerr=df_std[f"{var}"].values, color='blue', ecolor="lightgrey",
#                     label = f"{var}",  fmt='-o', capsize=1)
#         ax.tick_params(axis='y', labelsize=10)
#         ax.set_ylabel(f"{var}", fontsize = 12)
#         ax.legend(fontsize=14, frameon=False, loc = "upper left") #bbox_to_anchor=(.8, 0.8)
#         ax.set_ylim(0,1)
#         # if i == len(varlist)-1:
#         # if (row ==1&col==0)or(row ==2&col==1):
#         # if i==6 or i==7: #諦め
#         ax.tick_params(axis='x', labelsize=10)
#         plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
#         # else:
#         #     ax.tick_params(axis='x', labelsize=0)
#         if (row ==3)&(col==1):
#             fig.delaxes(ax)
#             # ax.set_visible(False)
#             # for spine in ax.spines.values():
#             #     spine.set_visible(False)
#         if i ==0:
#             ax.set_title(f"{reginame} partial correlaton")
#         # fig.delaxes(axes[3,1])
#     axes[3, 1].set_axis_off()
#     plt.tight_layout()
#     ### Export fig
#     out_dir_fig = out_dir + os.sep + "_png"
#     os.makedirs(out_dir_fig, exist_ok=True)
#     fig.savefig(out_dir_fig + os.sep + f"{reginame}_partial.png")
#     plt.close()
    
    

