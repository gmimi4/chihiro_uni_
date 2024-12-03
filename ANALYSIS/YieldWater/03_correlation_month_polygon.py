# -*- coding: utf-8 -*-
"""
Assigin risk var and month to polygon
"""
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import glob

# pp = "partial"
pp = "_pearson_detr_0_neg"
# csv_std_slp = rf"D:\Malaysia\02_Timeseries\YieldWater\03_var_variation\{pp}\std_slope.csv"
# region_shp = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
# out_dir = rf"D:\Malaysia\02_Timeseries\YieldWater\03_var_variation\{pp}\_shp"
csv_std_slp = f"/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/03_var_variation/{pp}/std_slope.csv"
region_shp = "/Volumes/SSD_2/Malaysia/Validation/1_Yield_doc/shp/region_slope_fin.shp"
out_dir = f"/Volumes/SSD_2/Malaysia/02_Timeseries/YieldWater/03_var_variation/{pp}/_shp"
os.makedirs(out_dir,exist_ok=True)


# varlist = ['rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD'] #'GOSIF', 
# month_calendar = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec" }

""" prepare polygon"""
gdf_region = gpd.read_file(region_shp) 
gdf_region = gdf_region.drop(columns=['NAME_1','ADM1_EN'])


## name dict for original and csv name
# region_name_ori = list(gdf_region["Name"].values)
# region_name_dic = {}
# for r in region_name_ori:
#     r_csv = r.replace(" ","")
#     region_name_dic[r_csv] = r


df_std_slp = pd.read_csv(csv_std_slp, index_col=0)

""" attach risk month and var to polygon"""
gdf_region_copy = gdf_region.copy()
gdf_region_copy = gdf_region_copy.set_index("Name")
gdf_region_copy["month"]=0
gdf_region_copy["var"] = ""

for regi, row in df_std_slp.iterrows():
    risk_var = row["var"]
    risk_month = row["month"]
    gdf_region_copy.at[regi, "month"] = int(risk_month)
    gdf_region_copy.at[regi, "var"] = risk_var

          
""" # Export"""
gdf_region_copy.to_file(out_dir + os.sep + f"region_var_month_{pp}.shp")



# """ Extract crutial month and var """
# def get_key(my_dict, val):
#     for key, value in my_dict.items():
#         if value == val:
#             return key
        
# gdf_region_copy = gdf_region.copy()
# gdf_region_copy = gdf_region_copy.set_index("Name")
# gdf_region_copy["month"]=0
# gdf_region_copy["var"] = ""

# for csvfile in csvs:
#     """ read rank csv"""
#     filename = os.path.basename(csvfile)[:-4]
#     reginame_csv = filename.split("_")[1]
#     reginame_revert = region_name_dic[reginame_csv]
    
#     df_csv = pd.read_csv(csvfile, index_col=0)
#     df_csv_abs = df_csv.abs()
    
#     """ find hihest pearson var and month"""
#     ## drop GOSIF
#     try:
#         df_csv_abs = df_csv_abs.drop("GOSIF", axis=1)
#     except:
#         pass
#         # print("pass")
    
#     if len(df_csv_abs.dropna(how="all")) == 0:
#         continue
#     else:
#         ### Find critical var name and month
#         criti_var = df_csv_abs.max().idxmax()
#         criti_month = df_csv_abs[criti_var].idxmax() #month str
#         criti_month = get_key(month_calendar, criti_month.split('_')[0])
        
#     gdf_region_copy.at[reginame_revert, "month"] = int(criti_month)
#     gdf_region_copy.at[reginame_revert, "var"] = criti_var

# gdf_region_copy.to_file(out_dir + os.sep + f"region_criticals_{pp}.shp")
