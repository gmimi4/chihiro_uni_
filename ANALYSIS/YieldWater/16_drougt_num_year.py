# -'- coding: utf-8 -'-
"""
Correlation between annual FFB and the num of drought months
Use regional mean values made in CCM analysis
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import fnmatch
from scipy.stats import pearsonr
os.chdir(r"C:\Users\chihiro\Desktop\Python\ANALYSIS\YieldWater")
import _yield_csv

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

valname = "num_drought"
# drought_thre = 100 #100 mmrain/month

# month_dic = {"DJF":[12,1,2],"MAM":[3,4,5],"JJA":[6,7,8],"SON":[9,10,11]}

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
        
        """ collect drought """
        seri_rain = df_regimean["rain"]
        seri_rain_drough = seri_rain[seri_rain < drought_thre]
        num_year = {}
        for yr in years_regi:
            # drough_yr = seri_rain_drough[seri_rain_drough.index.year==yr]
            start_date = f"{yr-1}-01-01"
            end_date = f"{yr}-12-31"
            drough_yr = seri_rain_drough[start_date:end_date]
            drough_num = len(drough_yr)
            num_year[yr]=drough_num
        
        df_num_year = pd.DataFrame.from_dict(num_year, orient="index")
        df_num_year.columns = [valname]
            
        """ regional correlation"""
        df_concat = pd.concat([seri_yield, df_num_year], axis=1)
        df_concat = df_concat.dropna()
        ### Calculate Pearson correlation between two columns
        try:
            corr_slp, p_value = pearsonr(df_concat[valname], df_concat[regi])
            ## plot
            plt.figure(figsize=(5, 5))
            plt.scatter(df_concat[valname], df_concat[regi], label=f'annaul FFB vs num of monthly drought')
            # Calculate the least squares fit (linear regression)
            slope, intercept = np.polyfit(df_concat[valname], df_concat[regi], 1)
            regression_line = slope * df_concat[valname]+ intercept
            plt.plot(df_concat[valname], regression_line, color='grey',linestyle='--', label='Least Squares Fit')
            plt.xlabel(valname, fontsize=16)
            plt.ylabel('annual FFB', fontsize=16)
            # plt.legend()
            plt.text(0.1,0.8,'$ r $=' + str(round(corr_slp, 4)),fontsize=14, transform=plt.gca().transAxes)
            plt.text(0.1,0.7,'$ p $=' + str(round(p_value, 4)),fontsize=14, transform=plt.gca().transAxes)
            plt.tight_layout()
            plt.show()
            plt.savefig(out_dir_fin + os.sep + f"numdrougt_vsFFB_{regi}_{drought_thre}.png")
            plt.close()
        except:
            slope, corr_slp, p_value = 0, 0, 999
        
        """ ## this is for num csv """
        df_num_year_T = df_num_year.T
        df_num_year_T = df_num_year_T.rename(index = {valname:regi})
        df_num_year_T["slope"] = slope
        df_num_year_T["r"] = corr_slp
        df_num_year_T["pval"] = p_value
        num_results.append(df_num_year_T)
        
        
        
    ### Export with sorting
    df_num_results = pd.concat(num_results)
    df_num_results.index = df_num_results.index.str.replace(" ","")
    
    df_num_results = df_num_results.reindex(df_sample.index)
    df_num_results.to_csv(out_dir_fin + os.sep +f"numdrougt_{drought_thre}.csv")
            

""" # find best correlation"""
csvs_correlation = []
for root, dirs, files in os.walk(out_dir): #root: subdir
    for file in files:
        if fnmatch.fnmatch(file, "numdrougt_*.csv"):
            file_path = os.path.join(root, file)
            csvs_correlation.append(file_path)

bests = []
negatives = []
positives = []
for regi in regions:
    regifile = regi.replace(" ","")
    corr_regi = []
    for csvfile in csvs_correlation:
        df_corr = pd.read_csv(csvfile, index_col=0)
        seri_corr_regi = df_corr.loc[regifile,:]
        df_corr_regi = seri_corr_regi.to_frame()
        df_corr_regi = df_corr_regi.T
        thre = os.path.basename(csvfile)[:-4].split("_")[1]
        df_corr_regi["thre"] = int(thre)
        corr_regi.append(df_corr_regi)
    df_correlation_regi = pd.concat(corr_regi)
    
    ### smallest pval -> significant p
    df_correlation_regi = df_correlation_regi.reset_index()
    df_correlation_regi_valid = df_correlation_regi[df_correlation_regi['pval']<0.1]
    if len(df_correlation_regi_valid)>0:
        max_index = df_correlation_regi_valid['thre'].idxmax() #for threhold
        true_thre = df_correlation_regi_valid.loc[max_index]["thre"]
        min_index = df_correlation_regi['pval'].idxmin() #least p
        df_best_regi = df_correlation_regi.loc[min_index]
        df_best_regi = df_best_regi.to_frame().T
        df_best_regi = df_best_regi.set_index("index")
        df_best_regi["true_thre"]=true_thre
        bests.append(df_best_regi)
    else:
        df_best_regi = df_correlation_regi.iloc[0,:].to_frame().T
        df_best_regi = df_best_regi.set_index("index")
        df_best_regi[:] =np.nan
        bests.append(df_best_regi)
    
    ### negative r and smallest pval (*meaning drought stress)
    df_correlation_neg = df_correlation_regi[df_correlation_regi.r<0]
    if len(df_correlation_neg)>0:
        min_index = df_correlation_neg['pval'].idxmin()
        df_best_regi = df_correlation_neg.loc[min_index]
        df_best_regi = df_best_regi.to_frame().T
        df_best_regi = df_best_regi.set_index("index")
        df_correlation_regi_valid = df_correlation_neg[df_correlation_neg['pval']<0.1]
        if len(df_correlation_regi_valid)>0:    
            max_index = df_correlation_regi_valid['thre'].idxmax() #for threhold
            true_thre = df_correlation_regi_valid.loc[max_index]["thre"]
            df_best_regi["true_thre"]=true_thre
        else:
            df_best_regi["true_thre"]=np.nan
        negatives.append(df_best_regi)
    else:
        df_correlation_neg = df_correlation_regi.iloc[0,:].to_frame().T
        df_best_regi = df_correlation_neg.set_index("index")
        df_best_regi[:] =np.nan
        negatives.append(df_best_regi)
    
    ### positive r and smallest pval (*meaning drought welcome)
    df_correlation_posi = df_correlation_regi[df_correlation_regi.r>0]
    if len(df_correlation_posi)>0:
        min_index = df_correlation_posi['pval'].idxmin()
        df_best_regi = df_correlation_posi.loc[min_index]
        df_best_regi = df_best_regi.to_frame().T
        df_best_regi = df_best_regi.set_index("index")
        df_correlation_regi_valid = df_correlation_posi[df_correlation_posi['pval']<0.1]
        if len(df_correlation_regi_valid)>0:    
            max_index = df_correlation_regi_valid['thre'].idxmax() #for threhold
            true_thre = df_correlation_regi_valid.loc[max_index]["thre"]
            df_best_regi["true_thre"]=true_thre
        else:
            df_best_regi["true_thre"]=np.nan
        positives.append(df_best_regi)
    else:
        df_correlation_posi = df_correlation_regi.iloc[0,:].to_frame().T
        df_best_regi = df_correlation_posi.set_index("index")
        df_best_regi[:] =np.nan
        positives.append(df_best_regi)
    
## Export
df_bests = pd.concat(bests)
df_bests = df_bests.reindex(df_sample.index)
df_bests.to_csv(out_dir + os.sep +f"best_numdrougt.csv")

df_bests = pd.concat(negatives)
df_bests = df_bests.reindex(df_sample.index)
df_bests.to_csv(out_dir + os.sep +f"negative_numdrougt.csv")

df_bests = pd.concat(positives)
df_bests = df_bests.reindex(df_sample.index)
df_bests.to_csv(out_dir + os.sep +f"positive_numdrougt.csv")
    







