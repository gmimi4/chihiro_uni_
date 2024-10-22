# -*- coding: utf-8 -*-
"""
Scoring vulnerability using annual yield and results of this assessment
- Yield trend
- Yield correlation with ESNO/IOD index
- variable cv, MK slope
- variable anomaly in ENSO/IOD
each element is scaled 0 -1
multiplied by weights
"""
import os
import pandas as pd
import geopandas as gpd
import glob
import numpy as np

from tqdm import tqdm
import scipy.stats
from scipy.stats import zscore
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import datetime
from statistics import mean
import pingouin as pg
os.chdir(r"C:\Users\chihiro\Desktop\Python\ANALYSIS\YieldWater")
import _csv_to_dataframe
import _yield_csv  


shp_region = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
enso_yield_dir = r"D:\Malaysia\02_Timeseries\YieldWater\06_correlation_timelag_ENSO\_partial" 
var_yield_dir = r"D:\Malaysia\02_Timeseries\YieldWater\01_correlation_timelag\_partial"
cv_MK_file = r"D:\Malaysia\02_Timeseries\YieldWater\03_var_variation\_partial\std_slope.csv"
var_anomaly_file = r"D:\Malaysia\02_Timeseries\YieldWater\05_var_variation_ENSO\_partial\anomalies_ENSO_IOD.csv"
out_dir = r"D:\Malaysia\02_Timeseries\YieldWater\07_score_annual_yield"

startdate = datetime.datetime(2002,1,1)
enddate = datetime.datetime(2023,12,1)

def scalecolumn(df,tarcol):
    df.loc[:,f"{tarcol}s"] = (df[tarcol] - df[tarcol].min()) / (df[tarcol].max() - df[tarcol].min())
    return df

    

""" Weights """
yield_trend_weight = 3 #yield slope
yield_enso_weight = 2 #enson index correlation
variable_weight = 1 #related to variables
 


""" Yield trend"""
gdf_region = gpd.read_file(shp_region)
df_yieldtrend = gdf_region[["Name","slope"]]
df_yieldtrend = scalecolumn(df_yieldtrend,"slope")
df_yieldtrend = df_yieldtrend.set_index("Name")

regions = list(df_yieldtrend.index)
regions_csv = [r.replace(" ","") for r in regions]

revert_names = {}
for regi in regions:
    regi_csv = regi.replace(" ","")
    revert_names[regi_csv] = regi


""" def scoring"""
def scoring(df, tarcol, weight, revert=True):
    # df = df_yieldtrend
    df_tar = df.loc[:,tarcol].to_frame()
    df_tar = scalecolumn(df_tar, tarcol)
    df_tar[f"{tarcol}_score"] = df_tar[f"{tarcol}s"]*weight
    if revert:
        df_tar.index = df_tar.index.map(revert_names)
    return df_tar[f"{tarcol}_score"].to_frame()  

### Scoring yield trend
score_yield_trend = scoring(df_yieldtrend, "slope", 3,revert=False)



""" Yield correlation with ESNO/IOD index"""
csvs_enso = glob.glob(enso_yield_dir + os.sep + "*.csv")

### find the severesst correlation
def num_min_max_sum(df, tarcol):
    df_valid = df[[tarcol]].dropna()
    df_valid_abs = df_valid.abs()
    colnum = len(df_valid)
    try:
        # min
        # min_ = np.min(df_valid) #やめる
        # max
        max_ = np.max(df_valid_abs)
        max_idx = df_valid_abs.idxmax().values[0]
        sign = np.sign(df_valid.at[max_idx,tarcol]) #ElNinoとの関係で使う
        max_sign = max_ * sign
        # sum
        if len(df_valid_abs) ==0:
            sum_ = np.nan
        else:
            sum_ = np.sum(df_valid_abs).values[0]
    except:
        max_sign, sum_ = np.nan, np.nan
    
    return [colnum, max_sign, sum_]
        
    
enso_yield_result = {}
iod_yield_result = {}
for region in regions_csv:
    try:
        csv_region = [c for c in csvs_enso if region in c][0]
    except:
        continue
    df_enso = pd.read_csv(csv_region)
    
    enso_list = num_min_max_sum(df_enso, "ENSO")
    iod_list = num_min_max_sum(df_enso, "IOD")
    
    enso_yield_result[region] = enso_list
    iod_yield_result[region] = iod_list


new_column_names = {0: 'num', 1:'max', 2: 'sum'}
df_enso_result = pd.DataFrame.from_dict(enso_yield_result).T
df_enso_result = df_enso_result.rename(columns={df_enso_result.columns[i]: new_column_names[i] for i in new_column_names})

df_iod_result = pd.DataFrame.from_dict(iod_yield_result).T
df_iod_result = df_iod_result.rename(columns={df_iod_result.columns[i]: new_column_names[i] for i in new_column_names})

### Scoring
score_enso_num = scoring(df_enso_result, "num", 2) #abs
# score_enso_min = scoring(df_enso_result, "min", 2)
score_enso_max = scoring(df_enso_result, "max", 2) #符号含めscaling済み
score_enso_sum = scoring(df_enso_result, "sum", 2) #abs

score_iod_num = scoring(df_iod_result, "num", 2)
# score_iod_min = scoring(df_iod_result, "min", 2)
score_iod_max = scoring(df_iod_result, "max", 2)
score_iod_sum = scoring(df_iod_result, "sum", 2)


""" variable cv, MKslope """
df_cvmk = pd.read_csv(cv_MK_file, index_col=0)

## critical month and var are used later
df_critical = df_cvmk[["month","var"]]
df_critical = df_critical.reset_index()
df_critical["newindex"] = df_critical["index"].str.replace(" ","")
df_critical_nospace = df_critical.set_index("newindex")


### Scoring
score_cv = scoring(df_cvmk, "std", 1,revert=False)
score_mk = scoring(df_cvmk, "MKslope", 1,revert=False) #符号含めscaling済み


""" Yield correlation with vars"""
csvs_vars = glob.glob(var_yield_dir + os.sep + "*_abs.csv") #_absだけど符号あり
region_var_csv = [os.path.basename(c).split("_")[1] for c in csvs_vars]

var_result = {}
for region in region_var_csv:
    csv_region = [c for c in csvs_vars if region in c][0]
    df_region = pd.read_csv(csv_region)
    ## find critical var in the region    
    criical_var_region = df_critical_nospace.at[region,"var"]
    try:
        var_result_list = num_min_max_sum(df_region, criical_var_region)
    except: #no valid var
        var_result_list = [0,np.nan,np.nan]
    var_result[region] = var_result_list
    

# new_column_names = {0: 'num', 1: 'min', 2: 'max', 3:'sum'}
df_var_result = pd.DataFrame.from_dict(var_result).T
df_var_result = df_var_result.rename(columns={df_var_result.columns[i]: new_column_names[i] for i in new_column_names})

    
### Scoring
score_var_num = scoring(df_var_result, "num", 1)
# score_var_min = scoring(df_var_result, "min", 1)
score_var_max = scoring(df_var_result, "max", 1)
score_var_sum = scoring(df_var_result, "sum", 1)


"""variable anomaly in ENSO/IOD """
df_anomaly = pd.read_csv(var_anomaly_file, index_col=0)
df_anomaly = df_anomaly[["anomalyENSO","anomalyIOD"]]
df_anomaly_abs = df_anomaly.abs()

### Scoring
score_enso_anomaly = scoring(df_anomaly_abs, "anomalyENSO", 1, revert=False)
score_iod_anomaly = scoring(df_anomaly_abs, "anomalyIOD", 1, revert=False)


# """ negative or positive contribution for scoring: 良い状態ほど高得点"""
# ## -1: 大きいほど悪いとき
# # score_yield_trend
score_enso_num = score_enso_num *(-1)
# score_enso_min
score_enso_max =  score_enso_max *(-1)
score_enso_sum = score_enso_sum  *(-1)
score_iod_num = score_iod_num *(-1)
# score_iod_min
score_iod_max = score_iod_max *(-1)
score_iod_sum = score_iod_sum *(-1)
score_cv = score_cv *(-1)
score_mk = score_mk *(-1)
score_var_num = score_var_num *(-1)
# score_var_min
score_var_max = score_var_max *(-1)
score_var_sum = score_var_sum *(-1)
score_enso_anomaly = score_enso_anomaly *(-1)
score_iod_anomaly = score_iod_anomaly *(-1)
   
                          
""" concat all scores: 良い状態ほど高得点""" 
df_score_concat = pd.concat([score_yield_trend, 
                             score_enso_num, 
                             # score_enso_min, 
                             score_enso_max, 
                             score_enso_sum,
                             score_iod_num,
                             # score_iod_min,
                             score_iod_max,
                             score_iod_sum,
                             score_cv,score_mk,
                             score_var_num,
                             # score_var_min,
                             score_var_max,
                             score_var_sum,
                             score_enso_anomaly,
                             score_iod_anomaly
                             ],
                            axis=1)

### SUM scores
df_score_concat["sum_scores"] = df_score_concat.sum(axis=1) #
## scale scores
df_score_concat_valid = df_score_concat[df_score_concat["slope_score"]>=0]
df_score_fin = scalecolumn(df_score_concat_valid,"sum_scores")
df_score_fin = df_score_fin["sum_scoress"].to_frame()
df_score_fin["score_rank"] = df_score_fin["sum_scoress"].rank(ascending=False)
df_score_fin = df_score_fin.reset_index().rename(columns={"index":"Name"})


""" join to geodataframe"""
gdf_region = gpd.read_file(shp_region)
gdf_region_merge = pd.merge(gdf_region, df_score_fin, on='Name', how='inner')
gdf_region_merge.to_file(out_dir + os.sep + "annual_scores.shp")
