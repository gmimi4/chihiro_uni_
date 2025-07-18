# -'- coding: utf-8 -'-
"""
partial correlation among variables
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import scipy.stats as stats
from matplotlib.colors import TwoSlopeNorm
import pingouin as pg

monthly_mean_dir = r"D:\Malaysia\02_Timeseries\CCM\01_region_mean\EVI_allage"
out_dir = r"D:\Malaysia\02_Timeseries\YieldWater\11_var_matrix\_palmall\_partial"
os.makedirs(out_dir, exist_ok=True)

months = [m+1 for m in range(12)]

## sample for region order
csv_sample = r"D:\Malaysia\02_Timeseries\YieldWater\sample_order.csv"
df_sample = pd.read_csv(csv_sample,index_col=0)
df_sample.index = df_sample.index.str.replace(" ", "", regex=False)

csvs = glob.glob(monthly_mean_dir+os.sep + "*.csv")


def rename_df(df):
    df.columns = [col.replace("ano", "") for col in df.columns]
    df = df.rename(columns={"rain":"Rain","temp":"Temp"})
    return df

def heatmap(cov_matrix,keyname):
    plt.figure(figsize=(10, 8))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    ax = sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, norm=norm,
                )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18) #rotation=45
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
    # plt.title("Covariance Matrix Heatmap", fontsize=16)
    # plt.show()
    plt.savefig(out_dir + os.sep + f"{regi}_{keyname}.png")
    plt.close()
    
    ## Export as csv
    cov_matrix.to_csv(out_dir + os.sep + f"{regi}_{keyname}.csv")

def partial_matrix(df):
    cov_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
    p_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
    for var1 in df.columns:
        for var2 in df.columns:
            if var1 == var2:
                cov_matrix.loc[var1, var2] = 1.0
            else:
                control_vars = [col for col in df.columns if col not in [var1, var2]]
                result = pg.partial_corr(data=df, x=var1, y=var2, covar=control_vars)
                cov_matrix.loc[var1, var2] = result['r'].values[0]
                p_matrix.loc[var1, var2] = result['p-val'].values[0]
    
    cov_matrix = cov_matrix.astype(float)
    p_matrix = p_matrix.astype(float)
    
    return cov_matrix, p_matrix
    
    
for csvfile in tqdm(csvs):
    regi = os.path.basename(csvfile)[:-4].split("_")[0]
    df_csv = pd.read_csv(csvfile, index_col=0, parse_dates=True)
    df_csv = df_csv.drop("EVI",axis=1)
    
    """ # remove monthyl mean as anomly"""
    df_csv_use = df_csv.copy()
    vars_list = df_csv_use.columns.tolist()
    
    for var in vars_list:
        df_var = df_csv_use.loc[:,var]
        df_csv_use[f"{var}ano"] = np.nan
        for m in months:
            specific_month_rows = df_var[df_var.index.month == m]
            monthly_mean = specific_month_rows.mean(skipna=True)
            df_csv_use.loc[specific_month_rows.index, f"{var}ano"] = specific_month_rows - monthly_mean

    """ # z scoring"""
    varsano_list = [v+"ano" for v in vars_list]
    df_csv_remove = df_csv_use.loc[:,varsano_list]
    df_csv_remove = df_csv_remove.dropna()
    df_csv_remove_z = df_csv_remove.apply(stats.zscore) # z of deseasonal's
    
    df_csv_drop = df_csv.dropna()
    df_csv_z = df_csv_drop.apply(stats.zscore) # z of oridata
    

    """ # compute matrix (pearson)"""
    # Compute the covariance matrix
    df_csv_remove_z = rename_df(df_csv_remove_z)
    # cov_matrix = df_csv_remove_z.cov()
    cov_matrix, p_matrix = partial_matrix(df_csv_remove_z)
    heatmap(cov_matrix,"ano") #mistook... 
    p_matrix.to_csv(out_dir + os.sep + f"{regi}_ano_pval.csv")
    
    df_csv_z = rename_df(df_csv_z)
    # cov_matrix = df_csv_z.cov()
    cov_matrix, p_matrix = partial_matrix(df_csv_z)
    heatmap(cov_matrix,"ori")
    p_matrix.to_csv(out_dir + os.sep + f"{regi}_ori_pval.csv")
    
    
    
    # plt.figure(figsize=(10, 8))
    # norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    # sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, norm=norm)
    # # plt.title("Covariance Matrix Heatmap", fontsize=16)
    # # plt.show()
    # plt.savefig(out_dir + os.sep + f"{regi}_ori.png")
    # plt.close()
    
    
    
    
    

    
    





           

