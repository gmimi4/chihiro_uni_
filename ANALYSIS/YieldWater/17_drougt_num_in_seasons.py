# -'- coding: utf-8 -'-
"""
Count months less than threshold rain
"""
import os
import pandas as pd
import numpy as np
import fnmatch
from tqdm import tqdm
import matplotlib.pyplot as plt
os.chdir(r"C:\Users\chihiro\Desktop\Python\ANALYSIS\YieldWater")
import _yield_csv
plt.rcParams['font.family'] = 'Times New Roman'

monthly_mean_dir = r"D:\Malaysia\02_Timeseries\CCM\01_region_mean\EVI"
out_dir = r"D:\Malaysia\02_Timeseries\YieldWater\10_drought_correlation\01_num_drought\01_detrendFFB_year"
# os.makedirs(out_dir, exist_ok=True)

# startdate = datetime.datetime(2002,1,1)
# enddate = datetime.datetime(2023,12,31)

df_yield, df_yield_z, df_yield_detr = _yield_csv.main()
nondata_region = list(df_yield_z[df_yield_z.isna().all(axis=1)].index)
nondata_region = nondata_region +["Maluku Utara"]
regions = df_yield_detr.index.tolist()
regions = [r for r in regions if r not in nondata_region]

## sample for region order
csv_sample = r"D:\Malaysia\02_Timeseries\YieldWater\03_var_variation\_pearson_neg\std_slope.csv"
df_sample = pd.read_csv(csv_sample,index_col=0)
df_sample.index = df_sample.index.str.replace(" ", "", regex=False)

month_dic = {"DJF":[12,1,2],"MAM":[3,4,5],"JJA":[6,7,8],"SON":[9,10,11]}
header_season = list(month_dic.keys())

valname = "num_drought"
# drought_thre = 100 #100 mmrain/month
    

for drought_thre in tqdm([100, 120, 140, 160, 180, 200]):
    out_dir_fin = out_dir + os.sep + f"threhold_{drought_thre}"
    os.makedirs(out_dir_fin, exist_ok=True)
    
    num_results = []
    for regi in regions:
        regifile = regi.replace(" ","")
        csv_regimean = monthly_mean_dir + os.sep + f"{regifile}_meanvals.csv"
        df_regimean = pd.read_csv(csv_regimean, index_col=0, parse_dates=True)
        
        """ collect yield """
        seri_yield = df_yield_detr.loc[regi,:]
        seri_yield.index = seri_yield.index.astype('int32')
        years_regi = seri_yield.index.tolist()
        
        """ collect drought in seasons"""
        seri_rain = df_regimean["rain"]
        seri_rain_drough = seri_rain[seri_rain < drought_thre]
        num_year = {}
        for yr in years_regi:
            # drough_yr = seri_rain_drough[seri_rain_drough.index.year==yr]
            start_date = f"{yr-1}-01-01"
            end_date = f"{yr}-12-31"
            drough_yr = seri_rain_drough[start_date:end_date]
            
            num_season = {}
            for season, mons in month_dic.items():
                drough_yr_seas = drough_yr[drough_yr.index.month.isin(mons)]
                drough_num = len(drough_yr_seas)
                num_season[season] = drough_num
            
            num_year[yr]=num_season
        
        df_num_year = pd.DataFrame.from_dict(num_year, orient="index")
        
        
        """ # make multi_index_columns """
        multis = []
        for yr, row in df_num_year.iterrows():
            multicolumns = pd.MultiIndex.from_product([[yr], header_season], names=["Year", "Season"])
            row_ = row.to_frame().T
            row_.index = row_.index.astype(str)
            row_.index = [index.replace(str(yr), regi) for index in row_.index]
            df_multi = pd.DataFrame(row_.values, index=row_.index, columns=multicolumns)
            multis.append(df_multi)
        
            
        """ concat and collect """
        df_concat = pd.concat(multis, axis=1)
        
        num_results.append(df_concat)
        
        
    
    ### Export with sorting
    df_num_results = pd.concat(num_results)
    df_num_results.index = df_num_results.index.str.replace(" ","")
    
    df_num_results = df_num_results.reindex(df_sample.index)
    df_num_results.to_csv(out_dir_fin + os.sep +f"numdrougt_season_{drought_thre}.csv")
    


""" # find num in seasons at best correlation"""
csvs_correlation = []
for root, dirs, files in os.walk(out_dir): #root: subdir
    for file in files:
        if fnmatch.fnmatch(file, "numdrougt_season*.csv"):
            file_path = os.path.join(root, file)
            csvs_correlation.append(file_path)


corr_regi = []
for regi in regions:
    regifile = regi.replace(" ","")
    # corr_regi = []
    for csvfile in csvs_correlation:
        thre = os.path.basename(csvfile)[:-4].split("_")[2]
        df_corr = pd.read_csv(csvfile, header=[0, 1], index_col=0)
        seri_corr_regi = df_corr.loc[regifile,:]
        df_corr_regi = seri_corr_regi.to_frame()
        df_corr_regi = df_corr_regi.T
        nun_seas ={}
        for sea in header_season:
            df_seas = df_corr_regi.xs(sea, axis=1, level="Season")
            num_seas = np.sum(df_seas.values)
            nun_seas[sea] = num_seas
        df_num_sea = pd.DataFrame.from_dict(nun_seas, orient="index").T
        df_num_sea.index = df_num_sea.index.astype(str)
        df_num_sea = df_num_sea.rename(index={"0":regi})
        df_num_sea["thre"] = int(thre)
        corr_regi.append(df_num_sea)
            

df_correlation_regi = pd.concat(corr_regi) #cannot sort as sample
df_correlation_regi.to_csv(out_dir + os.sep + "seasons_numdrougt.csv")
           

#-------------------------------
""" Plot num in seasons (まだ汚い)"""
#-------------------------------
csv_num_season = out_dir + os.sep + "seasons_numdrougt.csv"
df_numseas = pd.read_csv(csv_num_season, index_col=0)

thre_use = 100
df_numseas_use = df_numseas[df_numseas.thre==thre_use]
df_numseas_use = df_numseas_use.drop(["thre"], axis=1)

cols = df_numseas_use.columns
fig,ax = plt.subplots(figsize=(10, 50))
fig.subplots_adjust(hspace=0.5)  
for regi in regions:
    ser_regi = df_numseas_use.loc[regi,:]
    ax.scatter(ser_regi.index, ser_regi.values, label = f"{regi}")
    ax.set_ylabel("the num of dry months", fontsize = 14)
    ax.legend(fontsize=10, loc='lower center', bbox_to_anchor=(.85, 1.0),frameon=False)
    plt.tight_layout()
    