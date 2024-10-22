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
"""# Critical vars"""
## ほんとうは有意なvarがないところとパームがないところも分かるようにしたい
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

region_shp = r"D:\Malaysia\02_Timeseries\YieldWater\01_correlation_timelag\_partial\_shp\region_criticals_partial.shp"
yield_shp = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
out_dir = os.path.dirname(region_shp)

varlist = ['rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD'] #'GOSIF',
gdf_region = gpd.read_file(region_shp)
gdf_region_ex = gdf_region[gdf_region["var"].isin(varlist)] #for frame
gdf_yield = gpd.read_file(yield_shp)

### extract area which have palm
gdf_yield_valid = gdf_yield[gdf_yield["rank_yield"]>0]
## find no correlation region
gdf_yield_valid_regions = list(gdf_yield_valid.Name)
gdf_region_ex_region = list(gdf_region_ex.Name)
nocorre_regions = [r for r in gdf_yield_valid_regions if r not in gdf_region_ex_region] #Perak
## this polygon color as no correlation
gdf_perak = gdf_region[gdf_region["Name"].isin(nocorre_regions)]

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
gdf_perak.plot(ax=ax, hatch='///', facecolor='none', edgecolor='grey',)
                # legend_kwds={'label': "no correlation"}) #"orientation": "horizontal")
gdf_region.boundary.plot(ax=ax, edgecolor='black', linewidth=0.7) ## frame
legend_handles = [Patch(color=color_map[var], label=var) for var in color_map.keys()]
# ax.legend(handles=legend_handles, #title="Variables", 
#           loc='upper right',framealpha=0, bbox_to_anchor=(1.15, 1),prop={'size': 16}) #
# Adding two legends: one for hatching, one for the color map
hatch_patch = mpatches.Patch(facecolor='none', edgecolor='grey', hatch='///',label='no correlation')
# Combine both color and hatch legends
all_handles = legend_handles + [hatch_patch]
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

region_shp = r"D:\Malaysia\02_Timeseries\YieldWater\01_correlation_timelag\_partial\_shp\region_criticals_partial.shp"
yield_shp = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
out_dir = os.path.dirname(region_shp)

gdf_region = gpd.read_file(region_shp)
gdf_region_ex = gdf_region[gdf_region.month!=0] #for frame
gdf_yield = gpd.read_file(yield_shp)

### extract area which have palm
gdf_yield_valid = gdf_yield[gdf_yield["rank_yield"]>0]
## find no correlation region
gdf_yield_valid_regions = list(gdf_yield_valid.Name)
gdf_region_ex_region = list(gdf_region_ex.Name)
nocorre_regions = [r for r in gdf_yield_valid_regions if r not in gdf_region_ex_region] #Perak
## this polygon color as no correlation
gdf_perak = gdf_region[gdf_region["Name"].isin(nocorre_regions)]


### color barのサイズが変えられない諦め
fontname='Times New Roman'
f_size_title = 20
f_size = 16        
fig, ax = plt.subplots(figsize=(12,4))
plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
fig.tight_layout()
gdf_region_ex.plot(ax=ax, legend=True, column='month', cmap="twilight",)
                   # legend_kwds={'label': "Month"}) #"orientation": "horizontal"
gdf_perak.plot(ax=ax, hatch='///', facecolor='none', edgecolor='grey',)
                # legend_kwds={'label': "no correlation"}) #"orientation": "horizontal")
gdf_region.boundary.plot(ax=ax, edgecolor='black', linewidth=0.7) ## frame
colorbar_ax = ax.get_figure().axes[-1] #to get the last axis of the figure, it's the colorbar axes
colorbar_ax.set_title("month", size=16)
colorbar_ax.tick_params(labelsize=16)
# Adding two legends: one for hatching, one for the color map
hatch_patch = mpatches.Patch(facecolor='none', edgecolor='grey', hatch='///',label='no correlation')
legend1 = ax.legend(handles=[hatch_patch], loc='upper right', fontsize=f_size, frameon=False)
ax.add_artist(legend1)
filename = os.path.basename(region_shp)[:-4]
plt.savefig(out_dir + os.sep + f'{filename}_month.png', dpi=600)



