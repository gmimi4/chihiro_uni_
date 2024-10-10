# -*- coding: utf-8 -*-
"""
arrange dataframe from csv
"""
import os
import pandas as pd
import numpy as np

def main(csvfile):
    df_csv = pd.read_csv(csvfile, index_col = 'datetime',
                         parse_dates=['datetime'])
    
    filename = os.path.basename(csvfile)[:-4]
    
    """ # AMSREとAMSR2のギャップを補正する"""   
    pi_e = df_csv.loc[:,["SMDSCE", "VODDSCE"]]
    pi_2 = df_csv.loc[:,["SMDSC2", "VODDSC2"]]
    # calc median
    pi_e_med = pi_e.median()
    pi_2_med = pi_2.median()
    ratio_sm = pi_2_med.SMDSC2 /pi_e_med.SMDSCE
    ratio_vod = pi_2_med.VODDSC2 /pi_e_med.VODDSCE
    
    df_csv["SMDSCErev"] = df_csv["SMDSCE"] *ratio_sm
    df_csv["VODDSCErev"] = df_csv["VODDSCE"] *ratio_vod
    
    
    """ #SMとVODは平均にする"""    
    # try: #AMSREとAMSR2を一列にする
    df_csv["SM"] = df_csv[["SMDSCErev", "SMDSC2"]].mean(skipna=True, axis='columns')
    df_csv["VOD"] = df_csv[["VODDSCErev", "VODDSC2"]].mean(skipna=True, axis='columns')
    df_csv= df_csv.drop(["SMDSCE","SMDSC2","VODDSCE","VODDSC2", "SMDSCErev", "VODDSCErev"], axis=1)
    
    
    # """ # set period"""
    # df_csv = df_csv[((df_csv.index.year >= startyear) & (df_csv.index.year <= endyear)) ]
    
    """ # zscoring with all seson""" 
    vars_list = df_csv.columns.tolist()
    
    df_csv_z = df_csv.copy()
    for var in vars_list:
        df_csv_z[f"{var}z"] = np.nan
        df_var = df_csv_z.loc[:,var]
        var_mean = df_var.mean()
        var_std = df_var.std()            
        df_csv_z[f"{var}z"] = (df_csv_z[var]-var_mean)/var_std
    
    """ # Scaling 0-1""" ##
    for var in vars_list:
        df_csv_z[f"{var}s"] = np.nan
        var_max = df_csv_z[var].max()
        var_min = df_csv_z[var].min()
        df_csv_z[f"{var}s"] = (df_csv_z[var]-var_min)/(var_max - var_min)
    
    return df_csv_z



    



