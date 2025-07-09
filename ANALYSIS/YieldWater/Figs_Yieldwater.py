# -*- coding: utf-8 -*-
"""
Cartopy is in gdal_copy (spyder ok) and rasterio_copy2 (no spyder)
"""

import os
import rasterio
# import georaster
from rasterio.plot import show
from rasterio.windows import Window
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
import glob
# import matplotlib as mpl
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

# -------------------------------------
"""# positive negative tif (ElNino devi)"""
# -------------------------------------
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import os
import rasterio
from rasterio.plot import show
from matplotlib.colors import Normalize
from matplotlib import colorbar, colors
import numpy as np
import math
plt.rcParams['font.family'] = 'Times New Roman'

in_dir = r"F:\MAlaysia\ENSO\01_deviations\_allevents\_mosaic"
out_dir = in_dir + os.sep + "_png"
os.makedirs(out_dir,exist_ok=True)


tifs = glob.glob(in_dir + os.sep + "mosaic_maxdevi_*.tif") #mosaic_mindevi_
rain_tif = [t for t in tifs if "rain" in t ][0]
temp_tif = [t for t in tifs if "temp" in t][0]
VOD_tif = [t for t in tifs if "VOD" in t][0]
SM_tif = [t for t in tifs if "SM" in t][0]
# KBDI_tif = [t for t in tifs if "KBDI_importance" in t][0]
Eb_tif = [t for t in tifs if "Eb" in t][0]
Et_tif = [t for t in tifs if "Et" in t][0]
vpd_tif = [t for t in tifs if "VPD" in t][0]

var_dic = {
    "Precipitation":rain_tif,
            "Temperature": temp_tif,
            "VOD":VOD_tif,
            "Soil moisture":SM_tif,
            "Evaporation":Eb_tif,
            "Transpiration":Et_tif,
            "Vapor Pressure Deficit":vpd_tif
            }

var_list = [
    rain_tif, temp_tif, vpd_tif, Et_tif, Eb_tif, SM_tif, VOD_tif
            ]


""" # obtain data range """
def raster_range(tif):
    with rasterio.open(tif) as src:
        arr = src.read(1)
    minval = np.nanmin(arr)
    maxval = np.nanmax(arr)
    return minval, maxval
    

""" # plot """
abc = {0:"(a)",1:"(b)",2:"(c)",3:"(d)",4:"(e)",5:"(f)",6:"(g)"}
extent_lonlat = (93, 145, -12, 9)
crs_lonlat = ccrs.PlateCarree()
fontname='Times New Roman'
f_size_title = 20
f_size = 16
fig, axes = plt.subplots(2, 4, figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(wspace=0.1, hspace=0.1)
axes = axes.flatten()
for i, tif in enumerate(var_list):
    #
    ax = axes[i]
    #
    src = rasterio.open(tif)
    arr = src.read(1)
    threshold = 1e+38
    arr[np.abs(arr) > threshold] = np.nan
    arr_1d = np.ravel(arr)
    arr_1d_nan = arr_1d[~np.isnan(arr_1d)]
    # range_max = 0.1
    # range_min = 0 # for visualization
    # tif_range = raster_range(tif)
    # range_min = tif_range[0]
    # range_max = tif_range[1]    
    # fig.suptitle(f"{variable} relationship", fontsize=f_size_title, fontname=fontname)
    ax.set_extent(extent_lonlat, crs=crs_lonlat)
    # ax.set_title(f"{peri}", va='bottom', fontsize=f_size, fontname=fontname)
    try:
        range_min = np.percentile(arr_1d_nan, 5)
        range_max = np.percentile(arr_1d_nan, 95)
        cf =show(arr, ax=ax, transform=src.transform, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=range_min, vcenter=0., vmax=range_max)) #norm=colors.TwoSlopeNorm(0),  set zero as center
    except:
        range_min = arr_1d_nan.min()
        range_max = arr_1d_nan.max()
        # range_min = np.percentile(arr_1d_nan, 5)
        # range_max = np.percentile(arr_1d_nan, 95)
        cf =show(arr, ax=ax, transform=src.transform, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=range_min, vcenter=0., vmax=range_max))
    ax.coastlines(linewidth=.5)
    ax.add_feature(cfeature.BORDERS, linewidth=.5, linestyle='-', edgecolor='black')
    ax.set_xticks([])
    ax.set_yticks([])
    src.close()    
    cbar_ax = ax.inset_axes([0.2, -0.2, 0.6, 0.05])
    cbar = plt.colorbar(cf.get_images()[0], cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=f_size * 0.8)
    # plt.colorbar(cf.get_images()[0], cax=cbar_ax, orientation="horizontal", location='bottom',
    #              norm=colors.Normalize(vmin=range_min, vmax=range_max),
    #              )
    ax.text(
        0.02, -0.2, abc[i],  # Adjust position (0.02, 0.95) for top-left corner
        transform=ax.transAxes,
        fontsize=f_size,
        fontname=fontname,
        weight='bold',
        va='top',
        ha='left',
        color='black'
    )
    src.close()
# Turn off unused axes
for j in range(len(var_list), len(axes)):
    axes[j].axis('off')
fig.tight_layout()
outname = os.path.basename(tif)[:-4].split("_")[1]
plt.savefig(out_dir + os.sep + f'{outname}.png', dpi=600)
plt.show()

# -------------------------------------
"""# Map risk #original values"""
# -------------------------------------

import matplotlib.pyplot as plt
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm, rgb2hex


def plot_yield_anomaly(colname):
    # colname = "score_rank"
    gdf_region_ex = gdf_region[gdf_region["slope"] !=0]
    #
    cmap= "cividis"#"YlOrBr_r"#'YlOrBr'
    cmapget = matplotlib.colormaps.get_cmap(cmap)
    #
    fontname='Times New Roman'
    f_size_title = 20
    f_size = 16        
    fig, ax = plt.subplots(figsize=(10,4))
    plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
    ## grey no palm
    gdf_region.plot(ax=ax, legend=False, facecolor='white', edgecolor='black',linewidth=0.5)
    # gdf_region.boundary.plot(ax=ax, facecolor='gray', edgecolor='black', linewidth=0.7) 
    gdf_region_ex.plot(ax=ax, legend=False, column=colname, cmap=cmap,  edgecolor='black',linewidth=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8) #label='Risk rank'
    cbar.ax.yaxis.set_ticks_position('right')  # Move the ticks to the right side
    cbar.ax.text(1.5, 1.02, "Low Risk", ha='center', va='bottom', fontsize=12, transform=cbar.ax.transAxes)
    cbar.ax.text(1.5, -0.03, "High Risk", ha='center', va='top', fontsize=12, transform=cbar.ax.transAxes)
    fig.subplots_adjust(right=0.8)  # Adjust the right space to bring colorbar closer
    fig.tight_layout()
    plt.savefig(out_dir + os.sep + f'{filename}_map.png', dpi=600)
    plt.close()
    

## Set --------------------------------------

yeild_shp = r"D:\Malaysia\02_Timeseries\YieldWater\09_Strategy\05_risk_from_correlation\risk.shp"

out_dir = os.path.dirname(yeild_shp)
filename = os.path.basename(yeild_shp)[:-4]

gdf_region = gpd.read_file(yeild_shp)

plot_yield_anomaly("allrisk")


# -------------------------------------
"""# Map risk #Categorical (if only valid polygon exist)"""
# -------------------------------------

import matplotlib.pyplot as plt
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm, rgb2hex

gdf_frame = gpd.read_file(r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp")

def plot_yield_anomaly2(colname, gdf):

    num_colors = 7
    cmap= "cividis"#"YlOrBr_r"#'YlOrBr'
    cmapget = matplotlib.colormaps.get_cmap(cmap)
    color_indices = np.linspace(0, 1, num_colors)  # 10 evenly spaced values between 0 and 1
    colors = [cmapget(i) for i in color_indices]
    code_list = [mcolors.to_hex(color[:3]) for color in colors]
    cmap = ListedColormap(code_list)
    categorical_label = list(range(0,num_colors,1))
    #
    gdf[colname +"cate"] = pd.cut(gdf[colname], bins=num_colors, labels=categorical_label)
	#
    fontname='Times New Roman'
    f_size_title = 20
    f_size = 16        
    fig, ax = plt.subplots(figsize=(10,4))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    gdf_frame.plot(ax=ax, legend=False, facecolor='white', edgecolor='black',linewidth=0.5)
    gdf.plot(ax=ax, legend=False, column=colname +"cate", cmap=cmap,  edgecolor='black',linewidth=0.5)
    ax.tick_params(axis='both', labelsize=14) #lat lon size
    ### Set up color bar with correct boundaries and labels}
    # min_value = 1 #rankのとき
    # max_value = len(gdf_region_ex) #rankのとき
    min_value = gdf[colname].min()
    max_value = gdf[colname].max()
    bounds = list(np.linspace(min_value,max_value, num_colors+1)) 
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ### Add color bar to the plot
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8,) #label='Risk rank')
    cbar.set_ticks(bounds)
    cbar.set_ticklabels([f"{bound:.1f}" for bound in bounds])
    cbar.ax.text(1.5, 1.03, "Low Risk", ha='center', va='bottom', fontsize=14, transform=cbar.ax.transAxes)
    cbar.ax.text(1.5, -0.04, "High Risk", ha='center', va='top', fontsize=14, transform=cbar.ax.transAxes)
    cbar.ax.yaxis.set_ticks_position('right')  # Move the ticks to the right side
    cbar.ax.tick_params(labelsize=14)  # tick label size for colorbar
    fig.subplots_adjust(right=0.8)  # Adjust the right space to bring colorbar closer
    fig.tight_layout()
    plt.savefig(out_dir + os.sep + f'{filename}_catemap.png', dpi=600)
    plt.close()
    

## Set --------------------------------------
yeild_shp = r"D:\Malaysia\02_Timeseries\YieldWater\09_Strategy\06_risk_from_correlation_byseasons\stress.shp"

out_dir = os.path.dirname(yeild_shp)
filename = os.path.basename(yeild_shp)[:-4]
gdf_region = gpd.read_file(yeild_shp)

plot_yield_anomaly2("ensorank", gdf_region)

# -------------------------------------
"""# Map risk #Categorical (if no data polygon exist)"""
# -------------------------------------

import matplotlib.pyplot as plt
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm, rgb2hex


def plot_yield_anomaly(colname):
    # colname = "score_rank"
    # gdf_region_ex = gdf_region[gdf_region["month"] !=0]
    gdf_region_ex = gdf_region[gdf_region["slope"] !=0]
    #
    num_colors = 7
    cmap= "cividis_r"#"YlOrBr_r"#'YlOrBr'
    cmapget = matplotlib.colormaps.get_cmap(cmap)
    color_indices = np.linspace(0, 1, num_colors)  # 10 evenly spaced values between 0 and 1
    colors = [cmapget(i) for i in color_indices]
    code_list = [mcolors.to_hex(color[:3]) for color in colors]
    cmap = ListedColormap(code_list)
    categorical_label = list(range(0,num_colors,1))
    #
    gdf_region_ex[colname +"cate"] = pd.cut(gdf_region_ex[colname], bins=num_colors, labels=categorical_label)
	#
    fontname='Times New Roman'
    f_size_title = 20
    f_size = 16        
    fig, ax = plt.subplots(figsize=(10,4))
    plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
    ## grey no palm
    gdf_region.plot(ax=ax, legend=False, facecolor='white', edgecolor='black',linewidth=0.5)
    # gdf_region.boundary.plot(ax=ax, facecolor='gray', edgecolor='black', linewidth=0.7) 
    gdf_region_ex.plot(ax=ax, legend=False, column=colname +"cate", cmap=cmap,  edgecolor='black',linewidth=0.5)
    ### Set up color bar with correct boundaries and labels}
    # min_value = 1 #rankのとき
    # max_value = len(gdf_region_ex) #rankのとき
    min_value = gdf_region_ex[colname].min()
    max_value = gdf_region_ex[colname].max()
    bounds = list(np.linspace(min_value,max_value, num_colors+1)) 
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ### Add color bar to the plot
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8,) #label='Risk rank')
    cbar.set_ticks(bounds)
    cbar.set_ticklabels([f"{bound:.1f}" for bound in bounds])
    cbar.ax.text(1.5, 1.02, "High Risk", ha='center', va='bottom', fontsize=12, transform=cbar.ax.transAxes)
    cbar.ax.text(1.5, -0.03, "Low Risk", ha='center', va='top', fontsize=12, transform=cbar.ax.transAxes)
    # Adjust position of the colorbar to be closer to the plot
    cbar.ax.yaxis.set_ticks_position('right')  # Move the ticks to the right side
    fig.subplots_adjust(right=0.8)  # Adjust the right space to bring colorbar closer
    fig.tight_layout()
    plt.savefig(out_dir + os.sep + f'{filename}_catemap.png', dpi=600)
    plt.close()
    

## Set --------------------------------------
# yeild_shp = r"D:\Malaysia\02_Timeseries\YieldWater\09_Strategy\03_overall_risk\_pearson_detr_0_neg\risk.shp"
yeild_shp = r"D:\Malaysia\02_Timeseries\YieldWater\09_Strategy\05_risk_from_correlation\risk_convert.shp"

out_dir = os.path.dirname(yeild_shp)
filename = os.path.basename(yeild_shp)[:-4]
gdf_region = gpd.read_file(yeild_shp)

# plot_yield_anomaly("risk_rank")
plot_yield_anomaly("allrisk")

# -------------------------------------
"""# ENSO overall correlation"""
# -------------------------------------

import matplotlib.pyplot as plt
import os
import geopandas as gpd
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

yeild_shp = r"D:\Malaysia\02_Timeseries\YieldWater\09_Strategy\01_overall_correlation\_pearson_detr_0\overall_correlation_ensoiod.shp"
boarder_shp = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
filename = os.path.basename(yeild_shp)[:-4]
out_dir = os.path.dirname(yeild_shp)

gdf_region = gpd.read_file(yeild_shp)
gdf_frame = gpd.read_file(boarder_shp)

def format_lon(x, pos):
        return f"{abs(int(x))}°{'E' if x >= 0 else 'W'}"
def format_lat(y, pos):
    return f"{abs(int(y))}°{'N' if y >= 0 else 'S'}"

def plot_yield_anomaly(colname):
    # gdf_region_filer = gdf_region[gdf_region[colname]!=0]
    gdf_region_filer = gdf_region
    
    vmin = gdf_region_filer[colname].min()
    vmax = gdf_region_filer[colname].max()
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0) #oridata: vmin=-1.5, vmax=1.5,
    # fontname='Times New Roman'
    f_size_title = 20
    f_size = 20            

    ## ChatGPT
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fig.tight_layout()
    gdf_frame.plot(ax=ax, facecolor='none', edgecolor='black', hatch='///', linewidth=0.5)
    # Plot main layer with colorbar
    sm = gdf_region_filer.plot(
    column=colname, cmap='coolwarm', legend=True, ax=ax, edgecolor='black',
    linewidth=0.5, norm=norm,
    legend_kwds={'shrink': 0.7, 'aspect': 10, 'label': 'Average correlation coefficient',}
    )
    # Customize colorbar font size
    cbar = sm.get_figure().get_axes()[-1]
    cbar.tick_params(labelsize=f_size)
    cbar.set_ylabel("Average \n correlation coefficient", fontsize=f_size)
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.tick_params(labelsize=18)
    fig.tight_layout()
    plt.savefig(out_dir + os.sep + f'{colname}.png', dpi=400)
    plt.close()
    

plot_yield_anomaly("ENSO")
plot_yield_anomaly("IOD")

# -------------------------------------
"""# El Nino yield anomaly negative to positive"""
# -------------------------------------

import matplotlib.pyplot as plt
import os
import geopandas as gpd
import matplotlib.colors as mcolors

# yeild_shp = r"D:\Malaysia\02_Timeseries\YieldWater\04_yield_with_ENSO\ENSOIOD_yield_anomaly.shp"
yeild_shp = r"D:\Malaysia\02_Timeseries\YieldWater\04_yield_with_ENSO\_detr\ENSOIOD_yield_detr_anomaly.shp"
filename = os.path.basename(yeild_shp)[:-4]
out_dir = os.path.dirname(yeild_shp)

gdf_region = gpd.read_file(yeild_shp)

def plot_yield_anomaly(colname):
    gdf_region_filer = gdf_region[gdf_region[colname]!=0]
    
    vmin = gdf_region_filer[colname].min()
    vmax = gdf_region_filer[colname].max()
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0) #oridata: vmin=-1.5, vmax=1.5,
    fontname='Times New Roman'
    f_size_title = 20
    f_size = 16            
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
    fig.tight_layout()
    # plot colored
    gdf_region_filer.plot(column=colname, cmap='coolwarm', legend=True, ax=ax, edgecolor='black', 
                          linewidth=0.5, norm=norm, legend_kwds={'shrink': 0.7})
    # plot boarder
    gdf_region.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
    # ax.set_xticks([])
    # ax.set_yticks([])
    
    plt.savefig(out_dir + os.sep + f'{colname}.png', dpi=600)
    plt.close()
    

plot_yield_anomaly("anomalyENS")
plot_yield_anomaly("anomalyIOD")

# -------------------------------------
"""# Plot geopandas rank"""
# -------------------------------------
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_rank(region_shp,rankcol,filename, cmap):
    # region_shp = region_shp_dir + os.sep + f"sign_change_{var}_regionmean.shp" 
    out_dir = os.path.dirname(region_shp) + os.sep + "_png"
    os.makedirs(out_dir, exist_ok=True)
    
    gdf_region = gpd.read_file(region_shp)
    gdf_region_ex = gdf_region[gdf_region[rankcol] !=0]
    ### color barのサイズが変えられない諦め
    f_size_title = 20
    f_size = 16        
    fig, ax = plt.subplots(figsize=(10,5))
    plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
    fig.tight_layout()
    # cmap = "cividis_r"
    column = rankcol
    gdf_region_ex.plot(ax=ax, legend=True, column=column, cmap=cmap, #column="log"
                        legend_kwds={"location":"bottom","shrink":.6
                            # "orientation": "horizontal" #{'label': "Month"}
                            })
    gdf_region.boundary.plot(ax=ax, edgecolor='black', linewidth=0.7) ## frame
    
    # filename = os.path.basename(region_shp)[:-4]
    plt.savefig(out_dir + os.sep + f'{filename}.png', dpi=600)
    plt.clf()
    plt.close()
    
## Run
shp_path = r"D:\Malaysia\02_Timeseries\YieldWater\07_score_annual_yield\annual_scores.shp"
filename = os.path.basename(shp_path)[:-4] + "_scorerank.shp"
plot_rank(shp_path,"score_rank", filename, "cividis_r")

filename = os.path.basename(shp_path)[:-4] + "_score.shp"
plot_rank(shp_path,"sum_scores", filename, "cividis")


# -------------------------------------
"""# Stress"""
# -------------------------------------

import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import os
import numpy as np
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcdefaults() 

region_shp = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
stress_shp = r"D:\Malaysia\02_Timeseries\YieldWater\09_Strategy\06_risk_from_correlation_byseasons\stress.shp"
out_dir = os.path.dirname(stress_shp)

varlist = ['rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD'] #'GOSIF',
gdf_region = gpd.read_file(region_shp) #for frame

### extract area which have palm
gdf_stress = gpd.read_file(stress_shp)


color_map = {
    'Water deficit in leave': 'orange',     # Replace 'value_1' with your specific value and color
    'Available water deficit': 'yellow',    # Example: set color for 'value_2'
    'Excessive water in soil': 'purple',   # Example: set color for 'value_3'
    'Reduced dry effect': 'deepskyblue',   # Example: set color for 'value_4'
    'Lack of radiation':'lightgrey',}

gdf_stress['color'] = gdf_stress['ensostress'].map(color_map) 
     
fig, ax = plt.subplots(figsize=(16,4))
plt.subplots_adjust(wspace=0.1, hspace=0.1)
gdf_stress.plot(ax=ax, legend=True, color=gdf_stress['color'],)
gdf_region.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5) ## frame
legend_handles = [Patch(color=color_map[var], label=var) for var in color_map.keys()]
all_handles = legend_handles
ax.legend(handles=all_handles, loc='upper right', fontsize=16, frameon=False, bbox_to_anchor=(1.4, 1))
ax.tick_params(axis='both', labelsize=14)
fig.tight_layout()
filename = os.path.basename(stress_shp)[:-4]
plt.savefig(out_dir + os.sep + f'{filename}_stress.png', dpi=600)

# -------------------------------------
"""# Risk season"""
# -------------------------------------
import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
# from matplotlib.patches import Patch as mpatch
import matplotlib.patches as mpatches
import os
import numpy as np


# yield_shp = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
out_dir = os.path.dirname(region_shp)

gdf_region = gpd.read_file(region_shp)
gdf_region_ex = gdf_region[gdf_region.month!=0] #for frame

### extract area which have palm
gdf_yield_valid = gdf_region[gdf_region["rank_yield"]>0]
## find no correlation region
gdf_yield_valid_regions = list(gdf_yield_valid.Name)
gdf_region_ex_region = list(gdf_region_ex.Name)
nocorre_regions = [r for r in gdf_yield_valid_regions if r not in gdf_region_ex_region] #Perak
## this polygon color as no correlation
# gdf_perak = gdf_region[gdf_region["Name"].isin(nocorre_regions)]


### color barのサイズが変えられない諦め
fontname='Times New Roman'
f_size_title = 20
f_size = 16        
fig, ax = plt.subplots(figsize=(12,4))
plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
fig.tight_layout()
gdf_region_ex.plot(ax=ax, legend=True, column='month', cmap="twilight",)
                   # legend_kwds={'label': "Month"}) #"orientation": "horizontal"
# gdf_perak.plot(ax=ax, hatch='///', facecolor='none', edgecolor='grey',)
                # legend_kwds={'label': "no correlation"}) #"orientation": "horizontal")
gdf_region.boundary.plot(ax=ax, edgecolor='black', linewidth=0.7) ## frame
colorbar_ax = ax.get_figure().axes[-1] #to get the last axis of the figure, it's the colorbar axes
colorbar_ax.set_title("month", size=16)
colorbar_ax.tick_params(labelsize=16)
# hatch_patch = mpatches.Patch(facecolor='none', edgecolor='grey', hatch='///',label='no correlation')
# legend1 = ax.legend(handles=[hatch_patch], loc='upper right', fontsize=f_size, frameon=False)
# ax.add_artist(legend1)
filename = os.path.basename(region_shp)[:-4]
plt.savefig(out_dir + os.sep + f'{filename}_month.png', dpi=600)


# -------------------------------------
"""# Critical vars"""
# -------------------------------------

import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import os
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'

pp = "_pearson_detr_0_neg"
region_shp = rf"D:\Malaysia\02_Timeseries\YieldWater\03_var_variation\{pp}\_shp\region_var_month_{pp}.shp"
# yield_shp = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
out_dir = os.path.dirname(region_shp)

varlist = ['rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD'] #'GOSIF',
gdf_region = gpd.read_file(region_shp)
gdf_region_ex = gdf_region[gdf_region["var"].isin(varlist)] #for frame

### extract area which have palm
gdf_yield_valid = gdf_region[gdf_region["rank_yield"]>0]
## find no correlation region
gdf_yield_valid_regions = list(gdf_yield_valid.Name)
gdf_region_ex_region = list(gdf_region_ex.Name)
nocorre_regions = [r for r in gdf_yield_valid_regions if r not in gdf_region_ex_region] #Perak
## this polygon color as no correlation
# gdf_perak = gdf_region[gdf_region["Name"].isin(nocorre_regions)]

color_map = {
    'rain': 'skyblue',     # Replace 'value_1' with your specific value and color
    'temp': 'palevioletred',    # Example: set color for 'value_2'
    'VPD': 'pink',   # Example: set color for 'value_3'
    'Et': 'lawngreen',   # Example: set color for 'value_4'
    'Eb':'goldenrod',
    'SM':'sienna',
    'VOD':'darkslategray'}

gdf_region_ex['color'] = gdf_region_ex['var'].map(color_map) 

### color barのサイズが変えられない諦め
fontname='Times New Roman'
f_size_title = 20
f_size = 16        
fig, ax = plt.subplots(figsize=(14,4))
plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
fig.tight_layout()
gdf_region_ex.plot(ax=ax, legend=True, color=gdf_region_ex['color'],)
                   # legend_kwds={'label': "Month"}) #"orientation": "horizontal"
# gdf_perak.plot(ax=ax, hatch='///', facecolor='none', edgecolor='grey',)
#                 # legend_kwds={'label': "no correlation"}) #"orientation": "horizontal")
gdf_region.boundary.plot(ax=ax, edgecolor='black', linewidth=0.7) ## frame
legend_handles = [Patch(color=color_map[var], label=var) for var in color_map.keys()]
# hatch_patch = mpatches.Patch(facecolor='none', edgecolor='grey', hatch='///',label='no correlation')
all_handles = legend_handles #+ [hatch_patch]
ax.legend(handles=all_handles, loc='upper right', fontsize=f_size, frameon=False, bbox_to_anchor=(1.25, 1))
filename = os.path.basename(region_shp)[:-4]
plt.savefig(out_dir + os.sep + f'{filename}_vars.png', dpi=600)

# -------------------------------------
"""# Critical months"""
# -------------------------------------
## color cycle: https://www.yutaka-note.com/entry/matplotlib_color_cycle

import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
# from matplotlib.patches import Patch as mpatch
import matplotlib.patches as mpatches
import os
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'

pp = "_pearson_detr_0_neg"
region_shp = rf"D:\Malaysia\02_Timeseries\YieldWater\03_var_variation\{pp}\_shp\region_var_month_{pp}.shp"
# yield_shp = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
out_dir = os.path.dirname(region_shp)

gdf_region = gpd.read_file(region_shp)
gdf_region_ex = gdf_region[gdf_region.month!=0] #for frame

### extract area which have palm
gdf_yield_valid = gdf_region[gdf_region["rank_yield"]>0]
## find no correlation region
gdf_yield_valid_regions = list(gdf_yield_valid.Name)
gdf_region_ex_region = list(gdf_region_ex.Name)
nocorre_regions = [r for r in gdf_yield_valid_regions if r not in gdf_region_ex_region] #Perak
## this polygon color as no correlation
# gdf_perak = gdf_region[gdf_region["Name"].isin(nocorre_regions)]


### color barのサイズが変えられない諦め
fontname='Times New Roman'
f_size_title = 20
f_size = 16        
fig, ax = plt.subplots(figsize=(12,4))
plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
fig.tight_layout()
gdf_region_ex.plot(ax=ax, legend=True, column='month', cmap="twilight",)
                   # legend_kwds={'label': "Month"}) #"orientation": "horizontal"
# gdf_perak.plot(ax=ax, hatch='///', facecolor='none', edgecolor='grey',)
                # legend_kwds={'label': "no correlation"}) #"orientation": "horizontal")
gdf_region.boundary.plot(ax=ax, edgecolor='black', linewidth=0.7) ## frame
colorbar_ax = ax.get_figure().axes[-1] #to get the last axis of the figure, it's the colorbar axes
colorbar_ax.set_title("month", size=16)
colorbar_ax.tick_params(labelsize=16)
# hatch_patch = mpatches.Patch(facecolor='none', edgecolor='grey', hatch='///',label='no correlation')
# legend1 = ax.legend(handles=[hatch_patch], loc='upper right', fontsize=f_size, frameon=False)
# ax.add_artist(legend1)
filename = os.path.basename(region_shp)[:-4]
plt.savefig(out_dir + os.sep + f'{filename}_month.png', dpi=600)



