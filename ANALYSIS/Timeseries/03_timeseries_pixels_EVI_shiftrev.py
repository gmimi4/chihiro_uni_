# -*- coding: utf-8 -*-
"""
数ピクセルずれてるのをcsvを調整して直す
"""

import os, sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
    
in_parent_dir = '/Volumes/SSD_2/Malaysia/02_Timeseries/CPA_CPR/0_vars_timeseries/EVI'
out_dir = in_parent_dir + os.sep + "_shiftrev"
os.makedirs(out_dir, exist_ok=True)

varlist = ["rain", "temp", "Et", "Eb","VPD", "SMDSCE", "VODDSCE", "SMDSC2", "VODDSC2"]

shift_dir = {
    "A1":
        {"rain":-3,"temp":-1,"Et":-2,"Eb":-2,"VPD":-3,"SMDSCE":-3,"VODDSCE":-3,"SMDSC2":-2,"VODDSC2":-2},
    "A2":
        {"rain":0,"temp":0,"Et":0,"Eb":0,"VPD":0,"SMDSCE":0,"VODDSCE":0,"SMDSC2":0,"VODDSC2":0},
    "A3":
        {"rain":0,"temp":-1,"Et":0,"Eb":0,"VPD":-1,"SMDSCE":0,"VODDSCE":0,"SMDSC2":-1,"VODDSC2":-1},
    "A4":
        {"rain":1,"temp":1,"Et":1,"Eb":1,"VPD":1,"SMDSCE":1,"VODDSCE":1,"SMDSC2":1,"VODDSC2":1},
    }
        
       
for A in tqdm(["A1","A2","A3","A4"]):
    in_dir = in_parent_dir + os.sep + A
    csvs = glob.glob(in_dir + os.sep + "*.csv")
    for var in varlist:
        csvfile = [c for c in csvs if var in os.path.basename(c)][0]
        df = pd.read_csv(csvfile,index_col=0)
        
        """ shift dataset"""
        shiftnum = shift_dir[A][var]
        nanvals = [np.nan] * len(df.columns)
        nan_row = pd.DataFrame([nanvals], columns=df.columns, index=range(abs(shiftnum))) 
        if shiftnum >0:
            ## insert empty rows
            df_concat = pd.concat([nan_row, df])
            df_concat_rev = df_concat.reset_index(drop=True)
            ### cut end
            df_concat_fin = df_concat_rev.iloc[0:(shiftnum)*-1, :]
        elif shiftnum <0:
            ## delete first rows
            df_del = df.iloc[abs(shiftnum):, :]
            df_del = df_del.reset_index(drop=True)
            ## add nan rows in end
            df_concat = pd.concat([df_del, nan_row])
            df_concat_rev = df_concat.reset_index(drop=True)
            df_concat_fin = df_concat_rev
        else:
            df_concat_fin = df
        
        out_dir_A = out_dir + os.sep + A
        os.makedirs(out_dir_A, exist_ok=True)
        
        df_concat_fin.to_csv(out_dir_A + os.sep + os.path.basename(csvfile)[:-4]+"shift.csv")
            
            
            
    
        





