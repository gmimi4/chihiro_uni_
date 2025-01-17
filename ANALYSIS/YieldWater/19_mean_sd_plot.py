# -'- coding: utf-8 -'-
"""
correlation among variables
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob


monthly_mean_dir = r"D:\Malaysia\02_Timeseries\CCM\01_region_mean\EVI"
out_dir = r"D:\Malaysia\02_Timeseries\YieldWater\12_mean_sd_plot"
# os.makedirs(out_dir, exist_ok=True)

months = [m+1 for m in range(12)]

## sample for region order
csv_sample = r"D:\Malaysia\02_Timeseries\YieldWater\03_var_variation\_pearson_neg\std_slope.csv"
df_sample = pd.read_csv(csv_sample,index_col=0)
df_sample.index = df_sample.index.str.replace(" ", "", regex=False)

csvs = glob.glob(monthly_mean_dir+os.sep + "*.csv")

region_short = {
    "Johor":"Johor", "Kedah":"Kedah", "Kelantan":"Kelantan","Melaka":"Melaka",
    "NegeriSembilan":"Neger S","Pahang":"Pahang","Perak":"Perak","PulauPinang":"Pulau P",
    "Sabah":"Sabah", "Sarawak":"Sarawak","Selangor":"Selangor", "Terengganu":"Terengganu",
    "Aceh":"Aceh", "Banten":"Banten", "Bengkulu":"Bengkulu", "Gorontalo":"Gorontalo",
    "Jambi":"Jambi", "JawaBarat":"Jawa B", 
    "KalimantanBarat":"Kalim B","KalimantanSelatan":"Kalim S","KalimantanTengah":"Kalim Te", 
    "KalimantanTimur":"Kalim Ti", "KalimantanUtara":"Kalim U",
    "KepulauanBangkaBelitung":"Kepul BB", "KepulauanRiau":"Kepul R",
    "Lampung":"Lampung", "Maluku":"Maluku", "Papua":"Papua", "PapuaBarat":"Papua B",
    "Riau":"Riau", "SulawesiBarat":"Sulaw B", "SulawesiSelatan":"Sulaw S",
    "SulawesiTengah":"Sulaw Th", "SulawesiTenggara":"Sulaw Tr", 
    "SumateraBarat":"Sumat B", "SumateraSelatan":"Sumat S", "SumateraUtara":"Sumat U"
    }



var_list = ["rain","temp","VPD",	"Et","Eb","SM","VOD"]
    
for var in tqdm(var_list):
    result_regi={}
    for csvfile in csvs:
        regi = os.path.basename(csvfile)[:-4].split("_")[0]
        df_csv = pd.read_csv(csvfile, index_col=0, parse_dates=True)
        df_csv = df_csv.drop("EVI",axis=1)
        df_var = df_csv.loc[:,var]
        df_mean = df_var.mean()
        df_std = df_var.std()
        df_cv = df_std/df_mean
        result_regi[regi] = [df_mean, df_std, df_cv]
    
    df_result_var = pd.DataFrame.from_dict(result_regi,orient="index")
    df_result_var = df_result_var.rename(columns={0:"mean",1:"std",2:"cv"})
    
    df_result_var.to_csv(out_dir+os.sep+f"mean_std_{var}.csv")
    
    ## Plot #mean*std
    df_result_var.index = df_result_var.index.map(region_short)
    
    plt.figure(figsize=(12, 6))
    for reg in df_result_var.index:
        plt.scatter(df_result_var.loc[reg,"mean"], df_result_var.loc[reg,"std"], label=reg)        
    plt.legend(title="Regions", fontsize=10, title_fontsize=12,
               ncol=2,
               bbox_to_anchor=(1.05, 1), loc="upper left",borderaxespad=0) #左下を(0, 0), 右上を(1, 1)
    plt.xlabel("Mean", fontsize=16)
    plt.ylabel("Std", fontsize=16)
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_dir + os.sep + f"mean_std_{var}.png")
    plt.close()
    
    ## Plot #mean*std with text
    plt.figure(figsize=(12, 6))
    for reg in df_result_var.index:
        plt.scatter(df_result_var.loc[reg,"mean"], df_result_var.loc[reg,"std"], label=reg)
        texts = [plt.text(df_result_var.loc[reg,"mean"], df_result_var.loc[reg,"std"], reg, ha='center', va='center')]
    plt.legend(title="Regions", fontsize=10, title_fontsize=12,
               ncol=2,
               bbox_to_anchor=(1.05, 1), loc="upper left",borderaxespad=0) #左下を(0, 0), 右上を(1, 1)
    plt.xlabel("Mean", fontsize=16)
    plt.ylabel("Std", fontsize=16)
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_dir + os.sep + f"mean_std_text_{var}.png")
    plt.close()
    
    ## Plot #cv
    plt.figure(figsize=(15, 6))
    plt.bar(df_result_var.index, df_result_var["cv"], width=0.8)
    plt.xticks(rotation=90, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir + os.sep + f"cv_{var}.png")
    plt.close()
    
    
        
    
    
    
    
    
    
    

    
    





           

