# -*- coding: utf-8 -*-
"""
arrange yield csv to dataframe
produce df_yield and z scored df_yield
"""
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt

yield_csv_malay = r"D:\Malaysia\Validation\1_Yield_doc\Malaysia\Malaysia.csv"
yield_csv_indone = r"D:\Malaysia\Validation\1_Yield_doc\Indonesia\Indonesia_CPO.csv"

def main():
    """ prepare Yield df"""
    df_malay = pd.read_csv(yield_csv_malay, index_col=0)
    df_malay = df_malay.iloc[0:12,:]#delete unwanted rows
    df_indone = pd.read_csv(yield_csv_indone) #index_col=1
    df_indone = df_indone.iloc[:,1:] #delete unwanted column
    # rename 2 Nusa Tenggara
    df_indone.iat[17,0] = "Nusa Tenggara Barat"
    df_indone.iat[18,0] = "Nusa Tenggara Timur"
    # set index
    df_indone = df_indone.set_index('Unnamed: 1')
    # rename Bangka Belitung
    df_indone = df_indone.rename(index={'Bangka Belitung': 'Kepulauan Bangka Belitung'})

    # regions_indone = df_indone.index.tolist()

    ### convert ton/ha in Indone
    year_list_indone = df_indone.columns.tolist()
    year_list_indone = [y.split("_")[0] for y in year_list_indone]
    year_list_indone = sorted(list(set(year_list_indone)))

    ## From excel 
    # FFB = 4.1171*CPO + 447.06
    for yr in year_list_indone:
        ton_yr = df_indone[f"{yr}_ton"]
        ha_yr = df_indone[f"{yr}_ha"]
        ton_per_ha = (4.1171 * ton_yr) / ha_yr #FFB
        df_indone[f"{yr}"] = ton_per_ha #ton/ha

    df_indone = df_indone[year_list_indone]

    ## Combine Malaysia and Indonesia data
    df_all = pd.concat([df_malay, df_indone])

    ## eliminate outliers by quantile
    arr_all = df_all.values
    arr_all = np.ravel(arr_all)
    arr_all = arr_all[~np.isnan(arr_all)]

    def get_quantile(ar1d):
        quantile1 = np.percentile(ar1d, 25)
        quantile2 = np.percentile(ar1d, 50)
        quantile3 = np.percentile(ar1d, 75)
        
        iqr = quantile3 - quantile1
        lowval = quantile1 - iqr*1.5
        upperval = quantile3 + iqr*1.5
        
        return lowval, upperval

    ## remove outliers
    low_val, upper_val = get_quantile(arr_all)
    df_all_clean = df_all.mask(df_all > upper_val, np.nan)
    df_all_clean = df_all_clean.mask(df_all_clean < low_val, np.nan)
    df_yield = df_all_clean.sort_index(axis=1)
    
    """ z scoring yield"""
    df_yield_z = df_yield.copy() 
    for i, row in df_yield.iterrows():
        row_valid = row[~np.isnan(row)]
        reginame = row_valid.name
        years_row = row_valid.index.tolist()
        row_valid_z = zscore(row_valid, ddof=1) #ddof=0 標準偏差、ddof=1 不偏標準偏差
        for y in years_row:
            zs = row_valid_z.at[y]
            df_yield_z.at[reginame,y] = zs
    
    """ detrend yield """
    ### create least square line ###
    df_yield_detr = df_yield.copy()
    for i, row in df_yield_detr.iterrows():
        row_valid = row[~np.isnan(row)]
        df_row_valid = row_valid.to_frame()
        x = row_valid.index.astype("int")
        y = row_valid.values
        A = np.vstack([x, np.ones(len(x))]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0] #A, y
        
        df_row_valid['predict'] = a * df_row_valid.index.astype("int") + b
        df_row_valid['detr'] = df_row_valid[i] - df_row_valid['predict']
        row_valid_detr = df_row_valid['detr']
        row_valid_detr.name = i
        ### Scaling ###
        row_valid_detr_s =  (row_valid_detr - row_valid_detr.min())/(row_valid_detr.max()-row_valid_detr.min())
        for y in years_row:
            try:
                detr = row_valid_detr_s.at[y]
            except:
                detr = np.nan
            df_yield_detr.at[i,y] = detr
        
        ## check
        # regi = "Bengkulu"
        # x = df_yield.loc[regi,:].index.astype("int")
        # y = df_yield.loc[regi,:].values
        # plt.plot(x,y,"ro")
        # x = df_yield_detr.loc[regi,:].index.astype("int")
        # y = df_yield_detr.loc[regi,:].values
        # plt.plot(x,y,"bo")
        # plt.plot(x,(a*x+b),"g--")
        # plt.grid()
        # plt.show()
        # plt.close()

    
    return df_yield, df_yield_z, df_yield_detr



    



