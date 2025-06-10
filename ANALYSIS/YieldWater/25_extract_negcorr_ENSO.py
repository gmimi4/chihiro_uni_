# -*- coding: utf-8 -*-
"""
# extract climate correlation when negative corr with ENSO
"""
import os
import pandas as pd
import glob
import geopandas as gpd

use_corr = "all" #neg # select for target months when ENSO corr is neg or regardless corr
dir_enso = r"D:\Malaysia\02_Timeseries\YieldWater\06_correlation_timelag_ENSO\_pearson_detr_0"
pp = "_regionalmean"
dir_corr = rf"D:\Malaysia\02_Timeseries\YieldWater\01_correlation_timelag\{pp}"
dir_ano = r"D:\Malaysia\02_Timeseries\YieldWater\05_var_variation_ENSO\_composite_plot_years_regimean\_sig_anomaly"
dir_anolani = r"D:\Malaysia\02_Timeseries\YieldWater\05_var_variation_ENSO\_composite_plot_years_regimean_LaNina\_sig_anomaly"
dir_mean = r"D:\Malaysia\02_Timeseries\YieldWater\12_mean_sd_plot"
csv_sample = rf"D:\Malaysia\02_Timeseries\YieldWater\sample_order.csv"
out_dir = rf"D:\Malaysia\02_Timeseries\YieldWater\15_extract_corr_ENSO\{pp}\{use_corr}"
os.makedirs(out_dir, exist_ok=True)

csvs_enso = glob.glob(dir_enso + os.sep + "*.csv")
csvs_corr = glob.glob(dir_corr + os.sep + "*_abs.csv")
csvs_ano = glob.glob(dir_ano + os.sep + "*_ensoanomaly.csv")
csvs_anolani = glob.glob(dir_anolani + os.sep + "*_ensoanomaly.csv")
csvs_mean = glob.glob(dir_mean + os.sep + "mean_std_*.csv")

shp_region = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
gdf_region = gpd.read_file(shp_region) 
gdf_region = gdf_region.drop(columns=['NAME_1','ADM1_EN','slope','rank_yield'])
gdf_region["Name2"] = gdf_region["Name"].str.replace(" ","")
gdf_region_reset = gdf_region.set_index("Name2")

month_index = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12,
               "Jan_1":1,"Feb_1":2,"Mar_1":3,"Apr_1":4,"May_1":5,"Jun_1":6,"Jul_1":7,"Aug_1":8,"Sep_1":9,"Oct_1":10,"Nov_1":11,"Dec_1":12}

varlist = ["temp","rain","VPD", "Et","Eb","SM","VOD"]

# -------------------------------------
""" # First collect mean value for normalize anomaly"""
# -------------------------------------
mean_dic = {}
for var in varlist:
    csvf_mean = rf'D:\\Malaysia\\02_Timeseries\\YieldWater\\12_mean_sd_plot\\mean_std_{var}.csv'
    df_mean = pd.read_csv(csvf_mean, index_col=0)[["mean"]]
    mean_dic[var]=df_mean

# -------------------------------------
""" # Extract corr when significant ENSO impact"""
# -------------------------------------

dic_ano =  {"elnino":csvs_ano,"lanina":csvs_anolani}

for el, csvsano in dic_ano.items():
    overall = []
    for csvf_enso in csvs_enso:
        # csvf_enso = csvs_enso[8]
        region = os.path.basename(csvf_enso)[:-4].split("_")[1]
        csvf_corr = dir_corr + os.sep + f"peason_{region}_abs.csv"
        df_enso = pd.read_csv(csvf_enso, index_col=0)
        df_corr = pd.read_csv(csvf_corr, index_col=0) #monthly corr
        
        df_enso_sig = df_enso[df_enso.ENSO != 0]
        if len(df_enso_sig)==0:
            continue
        
        """ # select months for calc risk"""
        if use_corr =="neg":    
            """ # period when enso corr is significant"""
            index_sig = df_enso_sig.index.to_list() # obtain mon index
        if use_corr =="all":
            index_sig = list(month_index.keys()) #all 24 months
            
        """ # obtain climate corr for these index"""
        df_corr_sig = df_corr.loc[index_sig]
        # df_corr_sig.to_csv(out_dir + os.sep + f"{region}_sigenso.csv")
        
        # -------------------------------------
        """ # Extract anomly when negative ENSO impact"""
        # -------------------------------------
        csvf_ano = [c for c in csvsano if os.path.basename(c)[:-4].split("_")[0].replace(" ","")==region][0]
        df_ano = pd.read_csv(csvf_ano, index_col=0)
        index_sig_num = [month_index[m] for m in index_sig] # convert to numeric month
        index_sig_num = index_sig_num
        df_ano_sig = df_ano.loc[index_sig_num]
        # df_ano_sig.to_csv(out_dir + os.sep + f"{region}_sigenso.csv")
        
        # -------------------------------------
        """ # Calc impact like risk
            # climate corr * sig anomaly = negative is risk/impact """
        # -------------------------------------
        df_risk = df_corr_sig.copy(deep=True) # for final risk result
        df_risk[:] = 0
        for monstr, row in df_corr_sig.iterrows():
            mon_num = month_index[monstr]
            for var in varlist:
                anoval = df_ano_sig.loc[mon_num,f"{var}anovalid"]
                anoval_ratio = anoval / (mean_dic[var].loc[region,"mean"]) #make ratio
                corrval = row.loc[var] #row is series
                risk = anoval_ratio * corrval
                try:
                    df_risk.loc[monstr,var] = risk
                except:
                    print(region)
                    if len(risk)==2: #e.g. Jun and Jun_1 selected but anomaly is calculated only one Jun
                        risk = risk.iloc[0]
                        df_risk.loc[monstr,var] = risk
                    else:
                        print("something wrong")
        
        outname =  f"{region}_riskmatrix_{el}.csv"
        df_risk.to_csv(out_dir + os.sep + outname)
        
        df_risk_overall = df_risk.sum(axis=0).to_frame()
        df_risk_overall = df_risk_overall.T
        df_risk_overall = df_risk_overall.rename(index={0:region})
        overall.append(df_risk_overall)
            
            
    df_overall = pd.concat(overall)
    
    """ # calc risk proportion"""
    df_overall_copy = df_overall.copy(deep=True)
    df_overall_copy[df_overall_copy>0]=0
    df_overall_copy = df_overall_copy.abs()
    df_overall_copy_scale = df_overall_copy.apply(lambda x: (x-x.min())/ (x.max()-x.min()), axis=1)
    df_overall_copy_scale.columns = [f"{col}_prop" for col in df_overall_copy_scale.columns]
    df_overall_fin = pd.concat([df_overall, df_overall_copy_scale],axis=1)
    
    out_dir_overall = out_dir + os.sep + "_overall"
    os.makedirs(out_dir_overall, exist_ok=True)
    outname =  f"_overall_{el}.csv"
    df_overall_fin.to_csv(out_dir_overall + os.sep + outname)
    
    """ # join to shp """
    gdf_region_copy = gdf_region_reset.copy(deep=True)
    gdf_region_overall = pd.concat([gdf_region_copy, df_overall_fin], axis=1)
    outname =  f"_overall_{el}.shp"
    gdf_region_overall.to_file(out_dir_overall + os.sep + outname)
    

