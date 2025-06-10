# -*- coding: utf-8 -*-
"""
# Develop pie chart for overall risk
"""
import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

pp = "_regionalmean"
use_corr = "all" 
csv_overall_el = rf"D:\Malaysia\02_Timeseries\YieldWater\15_extract_corr_ENSO\{pp}\{use_corr}\_overall\_overall_elnino.csv"
csv_overall_la = rf"D:\Malaysia\02_Timeseries\YieldWater\15_extract_corr_ENSO\{pp}\{use_corr}\_overall\_overall_lanina.csv"
dir_ano_el = r"D:\Malaysia\02_Timeseries\YieldWater\05_var_variation_ENSO\_composite_plot_years_regimean\_sig_anomaly"
dir_ano_la = r"D:\Malaysia\02_Timeseries\YieldWater\05_var_variation_ENSO\_composite_plot_years_regimean_LaNina\_sig_anomaly"
out_dir = rf"D:\Malaysia\02_Timeseries\YieldWater\15_extract_corr_ENSO\{pp}\{use_corr}\_png"
os.makedirs(out_dir, exist_ok=True)

overall_dic = {"elnino":csv_overall_el, "lanina":csv_overall_la}

csvs_ano_el = glob.glob(dir_ano_el + os.sep + "*_ensoanomaly.csv")
csvs_ano_la = glob.glob(dir_ano_la + os.sep + "*_ensoanomaly.csv")
ano_dic = {"elnino":csvs_ano_el, "lanina":csvs_ano_la}

varlist = ["temp","rain","VPD", "Et","Eb","SM","VOD"]

colors = {"rain":'#5B9BD5', "temp":'#FFC000', "VPD":'#FFFF00', 
          "Et":'#00B050', "Eb":'#A6A6A6', "SM":'#1F4E79', "VOD":'#A9D18E'} #need make legend
hatch_pattern = '////'
def make_pichart(enso):
    csvf = overall_dic[enso] #risk calc by monthly corr * ano
    csvs_ano = ano_dic[enso] #for sign
    df = pd.read_csv(csvf, index_col=0)
    regions = df.index.to_list()    
    
    for regi in tqdm(regions):
        # regi = "Lampung"
        df_region = df.loc[regi,:]
        df_prop = df_region[df_region.index.str.contains('_prop')]
        df_prop = df_prop.replace(0, np.nan)
        df_prop = df_prop.dropna()
        var_use = df_prop.index.to_list()
        var_use = [v.replace("_prop","") for v in var_use]
        
        """ # get sign of anomaly of variable"""
        csv_ano_regi = [c for c in csvs_ano if os.path.basename(c).split("_")[0].replace(" ","")==regi][0]
        df_ano = pd.read_csv(csv_ano_regi, index_col=0)
        var_signs_dic = {}
        for var in var_use:
            df_ano_var = df_ano[f"{var}anovalid"]
            ano_mean = df_ano_var.mean() # sign of anomaly
            if ano_mean >0:
                var_signs_dic[var] = "I"
            if ano_mean <0:
                var_signs_dic[var] = "D" 
                
        var_signs = list(var_signs_dic.values()) #variable with D or I dictionary
        var_is = {i: v for i, v in enumerate(var_use)} # variable with index num dictionary
        """ # plot pie chart"""
        hatch_indices = []
        for i,sign in enumerate(var_signs):
            if sign =="D":
                hatch_indices.append(i)
        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, _ = ax.pie(df_prop.values, startangle=90, radius=1.0)    
        for w in wedges:
            w.set_label('')
        for i, wedge in enumerate(wedges):
            var_ = var_is[i]
            if i in hatch_indices:
                print(var_)
                wedge.set_facecolor('none')  # No fill
                wedge.set_edgecolor(colors[var_])  # Hatch color = edge color
                wedge.set_hatch(hatch_pattern)
            else:
                wedge.set_facecolor(colors[var_])
        ax.axis('equal')
        ax.set_title(regi, fontsize=50)
        # plt.show()
        out_enso_dir = out_dir + os.sep + enso
        os.makedirs(out_enso_dir, exist_ok=True)
        fig.savefig(out_enso_dir + os.sep + f"{regi}_pi_{enso}.png", dpi=600)
        plt.close()
        
        
enso = "elnino"
make_pichart(enso)      
enso = "lanina"
make_pichart(enso)

#----------------------------
""" # expot color legend"""
#----------------------------
import matplotlib.patches as mpatches

name_rev = {"rain":"Precipitation","temp":"Temperature","VPD":"VPD",
            "Et":"Transpiration","Eb":"Evaporation","SM":"Soil moisture","VOD":"VOD"}
colors_rev = {}
for name, cl in colors.items():
    colors_rev[name_rev[name]]=cl

handles = [mpatches.Patch(color=color, label=label) for label, color in colors_rev.items()]

fig, ax = plt.subplots(figsize=(6, 2))
ax.legend(handles=handles, loc='center', ncol=4, frameon=False)
ax.axis('off')
output_path = "/mnt/data/color_legend.png"
plt.savefig(out_dir + os.sep + "color_legend.png", bbox_inches='tight', dpi=300)


#----------------------------
""" # expot color legend"""
#----------------------------