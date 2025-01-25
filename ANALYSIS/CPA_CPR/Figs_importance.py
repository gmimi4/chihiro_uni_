# -*- coding: utf-8 -*-
"""
Cartopy is in gdal_copy (spyder ok) and rasterio_copy2 (no spyder)
"""

# -------------------------------------
"""# Plot geopandas rank"""
# -------------------------------------
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_rank(region_shp,rankcol):
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
    cmap = "YlGnBu"
    column = rankcol
    gdf_region_ex.plot(ax=ax, legend=True, column=column, cmap=cmap, #column="log"
                        legend_kwds={"location":"bottom","shrink":.6
                            # "orientation": "horizontal" #{'label': "Month"}
                            })
    gdf_region.boundary.plot(ax=ax, edgecolor='black', linewidth=0.7) ## frame
    
    filename = os.path.basename(region_shp)[:-4]
    plt.savefig(out_dir + os.sep + f'{filename}_rank.png', dpi=600)
    plt.clf()
    plt.close()
    
## Run
shp_path = r"D:\Malaysia\02_Timeseries\CPA_CPR\9_ISPRS_rank_correlation\ENSOdevi_mean_comprank.shp"
plot_rank(shp_path,"rank_other")
shp_path = r"D:\Malaysia\02_Timeseries\CPA_CPR\9_ISPRS_rank_correlation\sign_change_morethan1_regionmean_comprank.shp"
plot_rank(shp_path,"rank_other")
shp_path = r"F:\MAlaysia\Pegah-san\1_zonal\Suitmean2000-2017_mean.shp"
plot_rank(shp_path,"rank_suita") 


# -------------------------------------
"""# Plot geopandas stressed var""" ##離散色
# https://geodynamics.sci.hokudai.ac.jp/humet/meteo/tutorial.html
# https://hackmd.io/@h2tg95D2RP2ed-D8u-49Mg/S1moBqaRr
# -------------------------------------
import os
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm, rgb2hex
import numpy as np

varlist = ["rain","temp","VPD","Et","Eb","SM","VOD","morethan1"]

region_shp_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\8_ISPRS_stressed_region"
out_dir = region_shp_dir + os.sep + "_png"
os.makedirs(out_dir, exist_ok=True)

num_colors = 5
cmapget = matplotlib.colormaps.get_cmap('YlGnBu')
color_indices = np.linspace(0, 1, num_colors)  # 10 evenly spaced values between 0 and 1
colors = [cmapget(i) for i in color_indices]
code_list = [mcolors.to_hex(color[:3]) for color in colors]
cmap = ListedColormap(code_list)

for var in varlist:
    region_shp = region_shp_dir + os.sep + f"sign_change_{var}_regionmean.shp"
    # region_shp = r"D:\Malaysia\02_Timeseries\CPA_CPR\8_ISPRS_stressed_region\largechange_enso_1_75_regionmean.shp"
    gdf_region = gpd.read_file(region_shp)
    
    if var == "morethan1":
        var = "morethan"
    # var = "large"
    column = f"Ch{var}"
    gdf_region_ex = gdf_region[gdf_region[column] !=0]
    ### convert to categorical values
    categorical_label = list(range(0,num_colors,1))
    gdf_region_ex[column +"cate"] = pd.cut(gdf_region_ex[column], bins=num_colors, labels=categorical_label)
    f_size_title = 20
    f_size = 16        
    fig, ax = plt.subplots(figsize=(10,5))
    plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
    fig.tight_layout()
    gdf_region_ex.plot(ax=ax, legend=False, column=column +"cate", cmap=cmap, #column="log"
                        )
    ## frame
    gdf_region.boundary.plot(ax=ax, edgecolor='black', linewidth=0.7) 
    ### Set up color bar with correct boundaries and labels}
    unique_values = sorted(gdf_region_ex[column].unique())
    min_value = min(unique_values)
    max_value = max(unique_values)
    bounds = list(np.linspace(min_value,max_value, num_colors+1))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ### Add color bar to the plot
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.08, shrink=0.6)
    cbar.set_ticks(bounds)
    cbar.set_ticklabels([f"{bound:.2f}" for bound in bounds])
    
    filename = os.path.basename(region_shp)[:-4]
    plt.savefig(out_dir + os.sep + f'{filename}_categorical.png', dpi=600)
    plt.clf()
    plt.close()



# -------------------------------------
"""# Plot geopandas stressed var"""
# -------------------------------------
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl

varlist = ["rain","temp","VPD","Et","Eb","SM","VOD","morethan1"]

region_shp_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\8_ISPRS_stressed_region"
out_dir = region_shp_dir + os.sep + "_png"
os.makedirs(out_dir, exist_ok=True)

for var in varlist:
    region_shp = region_shp_dir + os.sep + f"sign_change_{var}_regionmean.shp"
    # region_shp = r"D:\Malaysia\02_Timeseries\CPA_CPR\8_ISPRS_stressed_region\largechange_enso_1_75_regionmean.shp"
    gdf_region = gpd.read_file(region_shp)
    
    if var == "morethan1":
        var = "morethan"
    # var = "large"
    gdf_region_ex = gdf_region[gdf_region[f"Ch{var}"] !=0]
    ### color barのサイズが変えられない諦め
    f_size_title = 20
    f_size = 16        
    fig, ax = plt.subplots(figsize=(10,5))
    plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
    fig.tight_layout()
    cmap = "YlGnBu"
    column = f"Ch{var}"
    gdf_region_ex.plot(ax=ax, legend=True, column=column, cmap=cmap, #column="log"
                        legend_kwds={"location":"bottom","shrink":.6
                            # "orientation": "horizontal" #{'label': "Month"}
                            })
    gdf_region.boundary.plot(ax=ax, edgecolor='black', linewidth=0.7) ## frame
    
    filename = os.path.basename(region_shp)[:-4]
    plt.savefig(out_dir + os.sep + f'{filename}.png', dpi=600)
    plt.clf()
    plt.close()


# -------------------------------------
"""# Final: large importance sign change (75%) and also enso decrease"""
## 離散色
# https://geodynamics.sci.hokudai.ac.jp/humet/meteo/tutorial.html
# https://hackmd.io/@h2tg95D2RP2ed-D8u-49Mg/S1moBqaRr
# -------------------------------------
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm, rgb2hex
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import os
import rasterio
from rasterio.plot import show
from matplotlib.colors import Normalize
from matplotlib import colorbar, colors
import numpy as np
import math
plt.rcParams['font.family'] = 'Times New Roman'

# tif = r"D:\Malaysia\02_Timeseries\CPA_CPR\4_stressed_pixels\timeeffect\sign_change_morethan1_filt.tif"
tif = r"D:\Malaysia\02_Timeseries\CPA_CPR\4_stressed_pixels\largechange_enso_1_75.tif"
out_dir = os.path.dirname(tif) + os.sep + "_png"

### Check arr_log or not!! below

""" # obtain data range """
def raster_range(tif):
    with rasterio.open(tif) as src:
        arr = src.read(1)
    minval = np.nanmin(arr)
    maxval = np.nanmax(arr)
    return minval, maxval
    

""" # plot """

extent_lonlat = (92, 147, -12, 9)
crs_lonlat = ccrs.PlateCarree()

fontname='Times New Roman'
f_size_title = 20
f_size = 16

      
fig, ax = plt.subplots(figsize=(10,5), subplot_kw={'projection': ccrs.PlateCarree()})
fig.tight_layout()
src = rasterio.open(tif)
arr = src.read(1)
arr_log = np.log10(arr)
arr_1d = np.ravel(arr)
arr_1d_log = np.log10(arr_1d)
# arr_1d_nan = arr_1d[~np.isnan(arr_1d)]
arr_1d_nan = arr_1d_log[~np.isnan(arr_1d_log)]
# range_min = arr_1d_nan.min()
# range_max = arr_1d_nan.max()
# range_max = 0.5
range_min = np.percentile(arr_1d_nan, 2)
range_max = np.percentile(arr_1d_nan, 98)
cmapget = matplotlib.colormaps.get_cmap('YlGnBu')
num_colors = 5
color_indices = np.linspace(0, 1, num_colors)  # 10 evenly spaced values between 0 and 1
colors = [cmapget(i) for i in color_indices]
code_list = [mcolors.to_hex(color[:3]) for color in colors]
cmap = ListedColormap(code_list)
## color range
bounds = list(np.linspace(range_min,range_max, num_colors+1)) #should be num_colors in boundaries
norm = BoundaryNorm(bounds,cmap.N)
ax.set_extent(extent_lonlat, crs=crs_lonlat)
cf =show(arr_log, ax=ax, transform=src.transform, cmap=cmap, norm=norm)
ax.coastlines(linewidth=.5)
ax.add_feature(cfeature.BORDERS, linewidth=.5, linestyle='-', edgecolor='black')
ax.set_xticks([])
ax.set_yticks([])
src.close()    
### set common color bar
cf_for_bar = cf.get_images()[0] #obtain image information
cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
plt.colorbar(cf_for_bar, ax=ax, cax=cbar_ax, orientation="horizontal", location='bottom')
plt.savefig(out_dir + os.sep + f'{os.path.basename(tif)[:-4]}_categorical.png', dpi=600)




# -------------------------------------
"""# Final: large importance sign change (75%) and also enso decrease"""
# -------------------------------------
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
import glob
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm, rgb2hex
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import os
import rasterio
from rasterio.plot import show
from matplotlib.colors import Normalize
from matplotlib import colorbar, colors
import numpy as np
import math
plt.rcParams['font.family'] = 'Times New Roman'

# tif = r"D:\Malaysia\02_Timeseries\CPA_CPR\4_stressed_pixels\timeeffect\sign_change_morethan1_filt.tif"
tif = r"D:\Malaysia\02_Timeseries\CPA_CPR\4_stressed_pixels\largechange_enso_1_75.tif"
out_dir = os.path.dirname(tif) + os.sep + "_png"

### Check arr_log or not!! below

""" # obtain data range """
def raster_range(tif):
    with rasterio.open(tif) as src:
        arr = src.read(1)
    minval = np.nanmin(arr)
    maxval = np.nanmax(arr)
    return minval, maxval
    

""" # plot """

extent_lonlat = (92, 147, -12, 9)
crs_lonlat = ccrs.PlateCarree()

fontname='Times New Roman'
f_size_title = 20
f_size = 16

      
fig, ax = plt.subplots(figsize=(10,5), subplot_kw={'projection': ccrs.PlateCarree()})
# plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
fig.tight_layout()

src = rasterio.open(tif)
arr = src.read(1)
arr_log = np.log10(arr)
arr_1d = np.ravel(arr)
arr_1d_log = np.log10(arr_1d)
# arr_1d_nan = arr_1d[~np.isnan(arr_1d)]
arr_1d_nan = arr_1d_log[~np.isnan(arr_1d_log)]

# range_min = arr_1d_nan.min()
# range_max = arr_1d_nan.max()
# range_max = 0.5
range_min = np.percentile(arr_1d_nan, 2)
range_max = np.percentile(arr_1d_nan, 98)
# range_min = 0 # for visualization

# fig.suptitle(f"relationshp change to stressed & El Nino damaged", fontsize=f_size_title, fontname=fontname)
ax.set_extent(extent_lonlat, crs=crs_lonlat)
# ax.set_title(f"Number of change variable: {i}", va='bottom', fontsize=f_size, fontname=fontname)

# norm = colors.Normalize(vmin=range_min, vmax=range_max)
cf =show(arr_log, ax=ax, transform=src.transform, cmap='YlGnBu', vmin=range_min, vmax=range_max) #norm = norm  #norm=colors.TwoSlopeNorm(0),  set zero as center
ax.coastlines(linewidth=.5)
ax.add_feature(cfeature.BORDERS, linewidth=.5, linestyle='-', edgecolor='black')
ax.set_xticks([])
ax.set_yticks([])

src.close()    

### set common color bar
cf_for_bar = cf.get_images()[0] #obtain image information
cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
plt.colorbar(cf_for_bar, ax=ax, cax=cbar_ax, orientation="horizontal", location='bottom',
             )

plt.savefig(out_dir + os.sep + f'{os.path.basename(tif)[:-4]}.png', dpi=600)


# -------------------------------------
"""# Multiple importance sign change (sum) and also enso decrease"""
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

# in_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\5_and_ENSO\timeeffect"
in_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\5_and_ENSO"
out_dir = in_dir + os.sep + "_png"

tifs = glob.glob(in_dir + os.sep + "*sign_change_morethan*enso.tif")

""" # obtain data range """
def raster_range(tif):
    with rasterio.open(tif) as src:
        arr = src.read(1)
    minval = np.nanmin(arr)
    maxval = np.nanmax(arr)
    return minval, maxval
    

""" # plot """

extent_lonlat = (92, 147, -12, 9)
crs_lonlat = ccrs.PlateCarree()

fontname='Times New Roman'
f_size_title = 20
f_size = 16

for var_tifs in tifs:
    #
    i = os.path.basename(var_tifs)[:-4].split("_")[-2][-1]
    #    
    fig, ax = plt.subplots(figsize=(10,5), subplot_kw={'projection': ccrs.PlateCarree()})
    # plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
    fig.tight_layout()
    #
    src = rasterio.open(var_tifs)
    arr = src.read(1)
    arr_log = np.log10(arr)
    arr_1d = np.ravel(arr)
    arr_1d_log = np.log10(arr_1d)
    # arr_1d_nan = arr_1d[~np.isnan(arr_1d)]
    arr_1d_nan = arr_1d_log[~np.isnan(arr_1d_log)]
    #
    # range_min = arr_1d_nan.min()
    # range_max = arr_1d_nan.max()
    # range_max = 0.5
    range_min = np.percentile(arr_1d_nan, 10)
    range_max = np.percentile(arr_1d_nan, 90)
    # range_min = 0 # for visualization
    #
    fig.suptitle(f"relationshp change to stressed & El Nino damaged", fontsize=f_size_title, fontname=fontname)
    ax.set_extent(extent_lonlat, crs=crs_lonlat)
    ax.set_title(f"Number of change variable: {i}", va='bottom', fontsize=f_size, fontname=fontname)
    #
    # norm = colors.Normalize(vmin=range_min, vmax=range_max)
    cf =show(arr_log, ax=ax, transform=src.transform, cmap='YlGnBu', vmin=range_min, vmax=range_max) #norm = norm  #norm=colors.TwoSlopeNorm(0),  set zero as center
    ax.coastlines(linewidth=.5)
    ax.add_feature(cfeature.BORDERS, linewidth=.5, linestyle='-', edgecolor='black')
    ax.set_xticks([])
    ax.set_yticks([])
    #
    src.close()    
    #
    ### set common color bar
    cf_for_bar = cf.get_images()[0] #obtain image information
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
    plt.colorbar(cf_for_bar, ax=ax, cax=cbar_ax, orientation="horizontal", location='bottom')
    #             )
    #
    plt.savefig(out_dir + os.sep + f'{os.path.basename(var_tifs)}.png', dpi=600)


# -------------------------------------
"""# Multiple importance sign change (sum) 離散色"""
# https://geodynamics.sci.hokudai.ac.jp/humet/meteo/tutorial.html
# https://hackmd.io/@h2tg95D2RP2ed-D8u-49Mg/S1moBqaRr
# -------------------------------------
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm, rgb2hex
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import os
import rasterio
from rasterio.plot import show
from matplotlib.colors import Normalize
from matplotlib import colorbar, colors
import numpy as np
import math
plt.rcParams['font.family'] = 'Times New Roman'


# in_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\4_stressed_pixels\timeeffect"
in_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\4_stressed_pixels"
out_dir = in_dir + os.sep + "_png"

tifs = glob.glob(in_dir + os.sep + "*sign_change_morethan*.tif")
num_vars = [os.path.basename(t)[:-4][-1] for t in tifs]


""" # obtain data range """
def raster_range(tif):
    with rasterio.open(tif) as src:
        arr = src.read(1)
    minval = np.nanmin(arr)
    maxval = np.nanmax(arr)
    return minval, maxval
    

""" # plot """

extent_lonlat = (92, 147, -12, 9)
crs_lonlat = ccrs.PlateCarree()

fontname='Times New Roman'
f_size_title = 20
f_size = 16


for i in num_vars:
    #
    var_tifs = [t for t in tifs if f"morethan{i}" in t][0]
    #    
    fig, ax = plt.subplots(figsize=(10,5), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.tight_layout()
    #
    src = rasterio.open(var_tifs)
    arr = src.read(1)
    arr_1d = np.ravel(arr)
    arr_log = np.log10(arr)
    arr_1d_log = np.log10(arr_1d)
    # arr_1d_nan = arr_1d[~np.isnan(arr_1d)]
    arr_1d_nan = arr_1d_log[~np.isnan(arr_1d_log)]
    #
    # range_min = arr_1d_nan.min()
    range_max = arr_1d_nan.max()
    # range_max = 0.5
    range_min = np.percentile(arr_1d_nan, 10)
    range_max = np.percentile(arr_1d_nan, 90)
    cmapget = matplotlib.colormaps.get_cmap('YlGnBu')
    num_colors = 5
    color_indices = np.linspace(0, 1, num_colors)  # 10 evenly spaced values between 0 and 1
    colors = [cmapget(i) for i in color_indices]
    code_list = [mcolors.to_hex(color[:3]) for color in colors]
    cmap = ListedColormap(code_list)
    ## color range
    bounds = list(np.linspace(range_min,range_max, num_colors+1)) #should be num_colors in boundaries
    norm = BoundaryNorm(bounds,cmap.N)
    #
    fig.suptitle(f"relationshp change to stressed", fontsize=f_size_title, fontname=fontname)
    ax.set_extent(extent_lonlat, crs=crs_lonlat)
    ax.set_title(f"Number of change variable: {i}", va='bottom', fontsize=f_size, fontname=fontname)
    #
    cf =show(arr_log, ax=ax, transform=src.transform, cmap=cmap, norm=norm)
    ax.coastlines(linewidth=.5) #YlGnBu #viridis #winter
    ax.add_feature(cfeature.BORDERS, linewidth=.5, linestyle='-', edgecolor='black')
    ax.set_xticks([])
    ax.set_yticks([])
    #
    src.close()    
    #
    ### set common color bar
    cf_for_bar = cf.get_images()[0] #obtain image information
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
    plt.colorbar(cf_for_bar, ax=ax, cax=cbar_ax, orientation="horizontal", location='bottom',)
    #
    plt.savefig(out_dir + os.sep + f'{os.path.basename(var_tifs)[:-4]}_categorical.png', dpi=600)

# -------------------------------------
"""# Multiple importance sign change (sum) 連続色"""
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


# in_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\4_stressed_pixels\timeeffect"
in_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\4_stressed_pixels"
out_dir = in_dir + os.sep + "_png"

tifs = glob.glob(in_dir + os.sep + "*sign_change_morethan*.tif")
num_vars = [os.path.basename(t)[:-4][-1] for t in tifs]


""" # obtain data range """
def raster_range(tif):
    with rasterio.open(tif) as src:
        arr = src.read(1)
    minval = np.nanmin(arr)
    maxval = np.nanmax(arr)
    return minval, maxval
    

""" # plot """

extent_lonlat = (92, 147, -12, 9)
crs_lonlat = ccrs.PlateCarree()

fontname='Times New Roman'
f_size_title = 20
f_size = 16

for i in num_vars:
    #
    var_tifs = [t for t in tifs if f"morethan{i}" in t][0]
    #    
    fig, ax = plt.subplots(figsize=(10,5), subplot_kw={'projection': ccrs.PlateCarree()})
    # plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
    fig.tight_layout()
    #
    src = rasterio.open(var_tifs)
    arr = src.read(1)
    arr_1d = np.ravel(arr)
    arr_log = np.log10(arr)
    arr_1d_log = np.log10(arr_1d)
    # arr_1d_nan = arr_1d[~np.isnan(arr_1d)]
    arr_1d_nan = arr_1d_log[~np.isnan(arr_1d_log)]
    #
    # range_min = arr_1d_nan.min()
    range_max = arr_1d_nan.max()
    # range_max = 0.5
    range_min = np.percentile(arr_1d_nan, 10)
    range_max = np.percentile(arr_1d_nan, 90)
    # range_min = 0 # for visualization
    #
    fig.suptitle(f"relationshp change to stressed", fontsize=f_size_title, fontname=fontname)
    ax.set_extent(extent_lonlat, crs=crs_lonlat)
    ax.set_title(f"Number of change variable: {i}", va='bottom', fontsize=f_size, fontname=fontname)
    #
    # norm = colors.Normalize(vmin=range_min, vmax=range_max) #cmap='cool' #viridis #YlGnBu
    cf =show(arr_log, ax=ax, transform=src.transform, cmap='YlGnBu', vmin=range_min, vmax=range_max) #norm = norm  #norm=colors.TwoSlopeNorm(0),  set zero as center
    ax.coastlines(linewidth=.5) #YlGnBu #viridis #winter
    ax.add_feature(cfeature.BORDERS, linewidth=.5, linestyle='-', edgecolor='black')
    ax.set_xticks([])
    ax.set_yticks([])
    #
    src.close()    
    #
    ### set common color bar
    cf_for_bar = cf.get_images()[0] #obtain image information
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
    plt.colorbar(cf_for_bar, ax=ax, cax=cbar_ax, orientation="horizontal", location='bottom',
                 )
    #
    plt.savefig(out_dir + os.sep + f'{os.path.basename(var_tifs)[:-4]}.png', dpi=600)

# -------------------------------------
"""# Importance sign change"""
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


in_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\4_stressed_pixels"
out_dir = in_dir + os.sep + "_png"


tifs = glob.glob(in_dir + os.sep + "*.tif")
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


""" # obtain data range """
def raster_range(tif):
    with rasterio.open(tif) as src:
        arr = src.read(1)
    minval = np.nanmin(arr)
    maxval = np.nanmax(arr)
    return minval, maxval
    

""" # plot """

extent_lonlat = (92, 147, -12, 9)
crs_lonlat = ccrs.PlateCarree()

fontname='Times New Roman'
f_size_title = 20
f_size = 16

for variable, var_tifs in var_dic.items():
    #
    fig, ax = plt.subplots(figsize=(10,5), subplot_kw={'projection': ccrs.PlateCarree()})
    # plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
    fig.tight_layout()
    #
    src = rasterio.open(var_tifs)
    arr = src.read(1)
    arr_log = np.log10(arr)
    arr_1d = np.ravel(arr)
    arr_1d_log = np.log10(arr_1d)
    # arr_1d_nan = arr_1d[~np.isnan(arr_1d)]
    arr_1d_nan = arr_1d_log[~np.isnan(arr_1d_log)]
    #
    # range_min = arr_1d_nan.min()
    # range_max = arr_1d_nan.max()
    range_min = np.percentile(arr_1d_nan, 10)
    range_max = np.percentile(arr_1d_nan, 90)
    # range_max = 0.1
    # range_min = 0 # for visualization
    #
    fig.suptitle(f"relationshp change to stressed", fontsize=f_size_title, fontname=fontname)
    ax.set_extent(extent_lonlat, crs=crs_lonlat)
    ax.set_title(f"{variable}", va='bottom', fontsize=f_size, fontname=fontname)
    #
    # norm = colors.Normalize(vmin=range_min, vmax=range_max)
    cf =show(arr_log, ax=ax, transform=src.transform, cmap='cool', vmin=range_min, vmax=range_max) #norm = norm  #norm=colors.TwoSlopeNorm(0),  set zero as center
    ax.coastlines(linewidth=.5)
    ax.add_feature(cfeature.BORDERS, linewidth=.5, linestyle='-', edgecolor='black')
    ax.set_xticks([])
    ax.set_yticks([])
    #
    src.close()    
    #
    ### set common color bar
    cf_for_bar = cf.get_images()[0] #obtain image information
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
    plt.colorbar(cf_for_bar, ax=ax, cax=cbar_ax, orientation="horizontal", location='bottom',
                 )
    #
    plt.savefig(out_dir + os.sep + f'{variable}_sign_chnange.png', dpi=600)

 

# -------------------------------------
"""# Major variable"""
# -------------------------------------
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
import glob
import matplotlib.pyplot as plt
from matplotlib import colorbar, colors
from matplotlib.colors import ListedColormap
import os
import rasterio
from rasterio.plot import show
import numpy as np

in_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\3_major_variable"
out_dir = in_dir

tifs = glob.glob(in_dir + os.sep + "*.tif")

extent_lonlat = (92, 147, -12, 9)
crs_lonlat = ccrs.PlateCarree()

fontname='Times New Roman'
f_size_title = 20
f_size = 16

categories =  {0: 'Nodata', 1: 'Precipitation', 2: 'Atomosphere (Temperature & VPD)',
               3: 'Vegetation (Trasnpiration & VOD)', 4: 'Soil (Evaporation & Soil moisture)'}
cmap = ListedColormap(['white','#069AF3', 'orangered', 'darkgreen', 'gold']) 
        
for tif in tifs:
    peri = os.path.basename(tif)[:-4].split("_")[-1]
    fig, ax = plt.subplots(figsize=(10,5), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
    fig.tight_layout()
       
    src = rasterio.open(tif)
    arr = src.read(1)
    
    fig.suptitle("Major variable", fontsize=f_size_title, fontname=fontname)
    ax.set_extent(extent_lonlat, crs=crs_lonlat)
    ax.set_title(peri, va='bottom', fontsize=f_size, fontname=fontname)
    
    # cf = show(src, ax=ax, transform=src.transform, cmap='viridis') #norm=colors.TwoSlopeNorm(0),  set zero as center
    cf = show(arr, ax=ax, transform=src.transform, cmap=cmap)
    ax.coastlines(linewidth=.5)
    ax.add_feature(cfeature.BORDERS, linewidth=.5, linestyle='-', edgecolor='black')
    ax.set_xticks([])
    ax.set_yticks([])
    
    src.close()    
        
    # Create a legend for the categories
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=categories[i]) for i in categories]
    ax.legend(handles=handles, loc='upper right') #title="Categories", 
    # fig.legend(prop={'family':fontname}) #Times new romanにならない 諦め    
    
    filename = os.path.basename(tif)[:-4]
    plt.savefig(out_dir + os.sep + f'{filename}.png', dpi=600)

# -------------------------------------
"""# Sensitivity change"""
# -------------------------------------
tif_sensitivity = r"D:\Malaysia\02_Timeseries\Sensitivity\1_sum\_mosaic\Diff_mosaic_sensitivity_sumcoef.tif"
out_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\1_sum\_mosaic\_png"
   
# period_str = os.path.basename(tif_sensitivity)[:-4].split("_")[-1]

extent_lonlat = (92, 147, -12, 9)
crs_lonlat = ccrs.PlateCarree()

fontname='Times New Roman'
f_size_title = 20
f_size = 16
        
fig, ax = plt.subplots(figsize=(10,5), subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
fig.tight_layout()
   
src = rasterio.open(tif_sensitivity)
arr = src.read(1)
arr_nan_2d = np.where(arr<-100, np.nan, arr)

## determin min and max of range
# tif_range = raster_range(tif_sensitivity)
arr_nan = arr[~np.isnan(arr)]
# range_min = np.percentile(arr_nan, 2)
# range_max = np.percentile(arr_nan, 98)
range_min = -1.5 #0.5
range_max = 1 #1.8


fig.suptitle("Sensitivity change", fontsize=f_size_title, fontname=fontname)
ax.set_extent(extent_lonlat, crs=crs_lonlat)
ax.set_title("2013-2022 - 2002-2012", va='bottom', fontsize=f_size, fontname=fontname)

# cf = show(src, ax=ax, transform=src.transform, cmap='viridis') #norm=colors.TwoSlopeNorm(0),  set zero as center
norm = colors.Normalize(vmin=range_min, vmax=range_max)
cf = show(arr_nan_2d, ax=ax, transform=src.transform, cmap='BrBG', norm=colors.TwoSlopeNorm(vmin=range_min, vcenter=0., vmax=range_max))
ax.coastlines(linewidth=.5)
ax.add_feature(cfeature.BORDERS, linewidth=.5, linestyle='-', edgecolor='black')
ax.set_xticks([])
ax.set_yticks([])

src.close()    
  
### set common color bar
base_ax = cf.get_images()[0]
cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
cbar = fig.colorbar(base_ax, ax=ax, cax=cbar_ax, orientation="horizontal", location='bottom',
             norm=colors.Normalize(vmin=range_min, vmax=range_max),
             )
# cbar.ax.tick_params(labelsize=6)

filename = os.path.basename(tif_sensitivity)[:-4]
plt.savefig(out_dir + os.sep + f'{filename}.png', dpi=600)

# -------------------------------------
"""# Sensitivity"""
# -------------------------------------
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from matplotlib import colorbar, colors
import os
import rasterio
from rasterio.plot import show
import numpy as np

in_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\1_sum\_mosaic"
out_dir = in_dir

tifs = glob.glob(in_dir + os.sep + "*.tif")

for tif_sensitivity in tifs:
    # tif_sensitivity = r"D:\Malaysia\02_Timeseries\Sensitivity\1_cv\2_cv_sensitivity\_mosaic\mosaic_sensitivity_cv_2002-2022.tif"
    
    period_str = os.path.basename(tif_sensitivity)[:-4].split("_")[-1]
    
    ## not used here
    # def raster_range(tif):
    #     with rasterio.open(tif) as src:
    #         arr = src.read(1)
    #     minval = np.nanmin(arr)
    #     maxval = np.nanmax(arr)
    #     return minval, maxval
    
    extent_lonlat = (92, 147, -12, 9)
    crs_lonlat = ccrs.PlateCarree()
    
    fontname='Times New Roman'
    f_size_title = 20
    f_size = 16
            
    fig, ax = plt.subplots(figsize=(10,5), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
    fig.tight_layout()
       
    src = rasterio.open(tif_sensitivity)
    arr = src.read(1)
    
    ## determin min and max of range
    # tif_range = raster_range(tif_sensitivity)
    arr_nan = arr[~np.isnan(arr)]
    # range_min = np.percentile(arr_nan, 2)
    # range_max = np.percentile(arr_nan, 98)
    range_min = 0.0 #0.5
    range_max = 2 #1.8
    
    
    fig.suptitle("Sensitivity", fontsize=f_size_title, fontname=fontname)
    ax.set_extent(extent_lonlat, crs=crs_lonlat)
    ax.set_title(period_str, va='bottom', fontsize=f_size, fontname=fontname)
    
    # cf = show(src, ax=ax, transform=src.transform, cmap='viridis') #norm=colors.TwoSlopeNorm(0),  set zero as center
    norm = colors.Normalize(vmin=range_min, vmax=range_max)
    cf = show(arr, ax=ax, transform=src.transform, cmap='viridis', norm=norm)
    ax.coastlines(linewidth=.5)
    ax.add_feature(cfeature.BORDERS, linewidth=.5, linestyle='-', edgecolor='black')
    ax.set_xticks([])
    ax.set_yticks([])
    
    src.close()    
      
    ### set common color bar
    base_ax = cf.get_images()[0]
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
    cbar = fig.colorbar(base_ax, ax=ax, cax=cbar_ax, orientation="horizontal", location='bottom',
                 norm=colors.Normalize(vmin=range_min, vmax=range_max),
                 )
    # cbar.ax.tick_params(labelsize=6)
    
    filename = os.path.basename(tif_sensitivity)[:-4]
    plt.savefig(out_dir + os.sep + f'{filename}.png', dpi=600)


# -------------------------------------
"""# CVs"""
# -------------------------------------

in_dir = r"D:\Malaysia\02_Timeseries\Sensitivity\1_cv\1_cv_ras\p_01\_mosaic"
out_dir = in_dir + os.sep + "_png"


tifs = glob.glob(in_dir + os.sep + "*.tif")
rain_tif = [t for t in tifs if "cv_rain" in t ]
temp_tif = [t for t in tifs if "cv_temp" in t]
VOD_tif = [t for t in tifs if "cv_VOD" in t]
SM_tif = [t for t in tifs if "cv_SM" in t]
# KBDI_tif = [t for t in tifs if "KBDI_importance" in t][0]
Eb_tif = [t for t in tifs if "cv_Eb" in t]
Et_tif = [t for t in tifs if "cv_Et" in t]
vpd_tif = [t for t in tifs if "cv_VPD" in t]
gosif_tif = [t for t in tifs if "cv_GOSIF" in t]

var_dic = {
    "Precipitation":rain_tif,
            "Temperature": temp_tif,
            "VOD":VOD_tif,
            "Soil moisture":SM_tif,
            "Evaporation":Eb_tif,
            "Transpiration":Et_tif,
            "Vapor Pressure Deficit":vpd_tif,
            "GOSIF":gosif_tif
            }


""" # obtain data range """
def raster_range(tif):
    with rasterio.open(tif) as src:
        arr = src.read(1)
        arr = np.where(arr<0, np.nan, arr)
    minval = np.nanmin(arr)
    maxval = np.nanmax(arr)
    return minval, maxval
    

""" # plot by year """

extent_lonlat = (92, 147, -12, 9)
crs_lonlat = ccrs.PlateCarree()

fontname='Times New Roman'
f_size_title = 20
f_size = 16

for variable, var_tifs in var_dic.items():
        
    fig, axs = plt.subplots(1, 1, figsize=(10,5), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
    fig.tight_layout()
    
    for peri in ["2002-2022"]: #"2002-2012", "2013-2022"
        
        tif = [t for t in var_tifs if peri in t][0]
        
        # tif_range = raster_range(tif)
        # range_min = tif_range[0]
        # range_max = tif_range[1]
        
        src = rasterio.open(tif)
        arr = src.read(1)
        arr = np.where(arr<0, np.nan, arr)
        
        arr_nan = arr[~np.isnan(arr)]
        range_min = np.percentile(arr_nan, 2)
        range_max = np.percentile(arr_nan, 98)
        
        
        ax = axs
        # if peri =="2002-2012":
        #     ax = axs[0]
        # if peri == "2013-2022":
        #     ax = axs[1]
        
        fig.suptitle(f"{variable} coefficient of variation (CV)", fontsize=f_size_title, fontname=fontname)
        ax.set_extent(extent_lonlat, crs=crs_lonlat)
        ax.set_title(f"{peri}", va='bottom', fontsize=f_size, fontname=fontname)
        
        norm = colors.Normalize(vmin=range_min, vmax=range_max)
        cf =show(arr, ax=ax, transform=src.transform, cmap='plasma', norm=norm) #norm=colors.TwoSlopeNorm(0),  set zero as center
        ax.coastlines(linewidth=.5)
        ax.add_feature(cfeature.BORDERS, linewidth=.5, linestyle='-', edgecolor='black')
        ax.set_xticks([])
        ax.set_yticks([])
        
        src.close()    
        
        # if peri == "2013-2022":
            ### Add a color bar #助かった：https://stackoverflow.com/questions/61327088/rio-plot-show-with-colorbar
            # cf_for_bar =ax.imshow(arr, cmap='coolwarm', alpha=0.0, extent=extent_lonlat, origin='upper',norm=colors.TwoSlopeNorm(vmin=range_min, vcenter=0., vmax=range_max)) #norm=colors.TwoSlopeNorm(0)
            # cf_for_bar = cf.get_images()[0] #obtain image information
            # cb = fig.colorbar(cf_for_bar, ax=ax, shrink=0.4, norm=colors.Normalize(vmin=range_min, vmax=range_max))
            # cb.ax.set_title("")
        # else:
        #     continue
            
    ### set common color bar
    cf_for_bar = cf.get_images()[0] #obtain image information
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
    plt.colorbar(cf_for_bar, ax=axs, cax=cbar_ax, orientation="horizontal", location='bottom',
                 norm=colors.Normalize(vmin=range_min, vmax=range_max),
                 )
    
    plt.savefig(out_dir + os.sep + f'{variable}_CV_2002-2022.png', dpi=600)

# -------------------------------------
"""# Relative importance"""
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

in_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_01\_mosaic"
out_dir = r"D:\Malaysia\02_Timeseries\CPA_CPR\2_out_ras\p_01\_mosaic\_png"


tifs = glob.glob(in_dir + os.sep + "*.tif")
rain_tif = [t for t in tifs if "rain_importance" in t ]
temp_tif = [t for t in tifs if "temp_importance" in t]
VOD_tif = [t for t in tifs if "VOD_importance" in t]
SM_tif = [t for t in tifs if "SM_importance" in t]
# KBDI_tif = [t for t in tifs if "KBDI_importance" in t][0]
Eb_tif = [t for t in tifs if "Eb_importance" in t]
Et_tif = [t for t in tifs if "Et_importance" in t]
vpd_tif = [t for t in tifs if "VPD_importance" in t]

var_dic = {
    "Precipitation":rain_tif,
            "Temperature": temp_tif,
            "VOD":VOD_tif,
            "Soil moisture":SM_tif,
            "Evaporation":Eb_tif,
            "Transpiration":Et_tif,
            "Vapor Pressure Deficit":vpd_tif
            }

rain_tif = [t for t in rain_tif if ("2002-2012" in t) or ("2013-2022" in t) ]
temp_tif = [t for t in temp_tif if ("2002-2012" in t) or ("2013-2022" in t) ]
VOD_tif = [t for t in VOD_tif if ("2002-2012" in t) or ("2013-2022" in t) ]
SM_tif = [t for t in SM_tif if ("2002-2012" in t) or ("2013-2022" in t) ]
Eb_tif = [t for t in Eb_tif if ("2002-2012" in t) or ("2013-2022" in t) ]
Et_tif = [t for t in Et_tif if ("2002-2012" in t) or ("2013-2022" in t) ]
vpd_tif = [t for t in vpd_tif if ("2002-2012" in t) or ("2013-2022" in t) ]


""" # obtain data range """
def raster_range(tif):
    with rasterio.open(tif) as src:
        arr = src.read(1)
    minval = np.nanmin(arr)
    maxval = np.nanmax(arr)
    return minval, maxval
    

""" # plot 2002-2012 & 2013-2022 """

extent_lonlat = (92, 147, -12, 9)
crs_lonlat = ccrs.PlateCarree()

fontname='Times New Roman'
f_size_title = 20
f_size = 16

for variable, var_tifs in var_dic.items():
        
    fig, axs = plt.subplots(1, 2, figsize=(15,5), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
    fig.tight_layout()
    
    for peri in ["2002-2012", "2013-2022"]:
        
        tif = [t for t in var_tifs if peri in t][0]
        
        tif_range = raster_range(tif)
        # range_min = tif_range[0]
        # range_max = tif_range[1]
        range_min = -0.7
        range_max = 0.7
        
        src = rasterio.open(tif)
        arr = src.read(1)
        
        if peri =="2002-2012":
            ax = axs[0]
        if peri == "2013-2022":
            ax = axs[1]
        
        fig.suptitle(f"{variable} relationship", fontsize=f_size_title, fontname=fontname)
        ax.set_extent(extent_lonlat, crs=crs_lonlat)
        ax.set_title(f"{peri}", va='bottom', fontsize=f_size, fontname=fontname)
        
        cf =show(arr, ax=ax, transform=src.transform, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=range_min, vcenter=0., vmax=range_max)) #norm=colors.TwoSlopeNorm(0),  set zero as center
        ax.coastlines(linewidth=.5)
        ax.add_feature(cfeature.BORDERS, linewidth=.5, linestyle='-', edgecolor='black')
        ax.set_xticks([])
        ax.set_yticks([])
        
        src.close()    
        
        # if peri == "2013-2022":
            ### Add a color bar #助かった：https://stackoverflow.com/questions/61327088/rio-plot-show-with-colorbar
            # cf_for_bar =ax.imshow(arr, cmap='coolwarm', alpha=0.0, extent=extent_lonlat, origin='upper',norm=colors.TwoSlopeNorm(vmin=range_min, vcenter=0., vmax=range_max)) #norm=colors.TwoSlopeNorm(0)
            # cf_for_bar = cf.get_images()[0] #obtain image information
            # cb = fig.colorbar(cf_for_bar, ax=ax, shrink=0.4, norm=colors.Normalize(vmin=range_min, vmax=range_max))
            # cb.ax.set_title("")
        # else:
        #     continue
            
    ### set common color bar
    cf_for_bar = cf.get_images()[0] #obtain image information
    cbar_ax = fig.add_axes([0.25, 0.15, 0.5, 0.02])
    plt.colorbar(cf_for_bar, ax=axs, cax=cbar_ax, orientation="horizontal", location='bottom',
                 norm=colors.Normalize(vmin=range_min, vmax=range_max),
                 )
    
    plt.savefig(out_dir + os.sep + f'{variable}_importance_chnange.png', dpi=600)



""" # Combine pngs """
pngs = glob.glob(in_dir + os.sep + "*.png")
rain_png = [p for p in pngs if "Precipitation" in p][0]
temp_png = [p for p in pngs if "Temperature" in p][0]
eb_png = [p for p in pngs if "Evaporation" in p][0]
et_png = [p for p in pngs if "Transpiration" in p][0]
sm_png = [p for p in pngs if "Soil Moisture" in p][0]
vod_png = [p for p in pngs if "VOD" in p][0]
kbdi_png = [p for p in pngs if "KBDI" in p][0]
blank_png = [p for p in pngs if "blank" in p][0]

# rain_im = Image.open(rain_png)
marginh, marginw = 400, 200
def crop_image(image_path, marginh, marginw):
    # Open the image
    image = Image.open(image_path)
    # Get image dimensions
    width, height = image.size
    # Calculate crop box coordinates
    left = marginw
    upper = marginh*0.8
    right = width - marginw*2.8
    lower = height - marginh*1.3
    # Crop the image
    cropped_image = image.crop((left, upper, right, lower))

    return cropped_image

# Example usage 
# cropped_image = crop_image(rain_png, 400, 200)
# kbdi_im.show()
rain_im = crop_image(rain_png, 400, 200)
temp_im = crop_image(temp_png, 400, 200)
eb_im = crop_image(eb_png, 400, 200)
et_im = crop_image(et_png, 400, 200)
sm_im = crop_image(sm_png, 400, 200)
vod_im = crop_image(vod_png, 400, 200)
kbdi_im = crop_image(kbdi_png, 400, 200)
blank_im = crop_image(blank_png, 400, 200)

 
#横に連結
for i, li in enumerate([[rain_im, temp_im], [eb_im, et_im], [sm_im, vod_im], [kbdi_im, blank_im]]):
    im1 = li[0]
    im2 = li[1]
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    dst.save(in_dir + os.sep + f'tmp{i}.png')

#縦に連結
impngs = glob.glob(in_dir + os.sep + "tmp*.png")
im1 = Image.open(impngs[0])
im2 = Image.open(impngs[1])
im3 = Image.open(impngs[2])
im4 = Image.open(impngs[3])


dst = Image.new('RGB', (im1.width, im1.height + im2.height+ im3.height+ im4.height))
dst.paste(im1, (0, 0))
dst.paste(im2, (0, im1.height))
dst.paste(im3, (0, im1.height+im2.height))
dst.paste(im4, (0, im1.height+im2.height+im3.height))
dst.paste(blank_im, (im1.width, im1.height+im2.height+im3.height))
dst.show()
dst.save(in_dir + os.sep + f'Importance_concat.png')

for t in impngs:
    os.remove(t)



# -------------------------------------
"""#Mann Kendall"""
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

tif_tempe = r"F:\MAlaysia\MannKendall\MannKendall_ERA1940_p0.05_2000.tif"
tif_rain = r"F:\MAlaysia\MannKendall\MannKendall_GOSIF_adj_yearly_p0.05.tif"
out_dir = r"F:\MAlaysia\SIF\GOSIF\02_tif_age_adjusted"

""" # obtain data range """
def raster_range(tif):
    with rasterio.open(tif) as src:
        arr = src.read(1)
    minval = np.nanmin(arr)
    maxval = np.nanmax(arr)
    return minval, maxval
    

""" # plot 2002-2012 & 2013-2022 """

extent_lonlat = (92, 147, -12, 9)
crs_lonlat = ccrs.PlateCarree()

fontname='Times New Roman'
f_size_title = 20
f_size = 16
        
fig, ax = plt.subplots(1,1, figsize=(10,5), subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
fig.tight_layout()

tif = tif_rain

tif_range = raster_range(tif)
# range_min = tif_range[0]
# range_max = tif_range[1]
with rasterio.open(tif) as src:
    arr = src.read(1)
    arr_ = arr[~np.isnan(arr)]
range_min = np.percentile(arr_, 2)
range_max = np.percentile(arr_, 98)

src = rasterio.open(tif)
arr = src.read(1)

# if peri =="2002-2012":
#     ax = axs[0]
# if peri == "2013-2022":
#     ax = axs[1]

fig.suptitle(f"GOSIF yearly trend by MannKendall", fontsize=f_size_title, fontname=fontname)
ax.set_extent(extent_lonlat, crs=crs_lonlat)
# ax.set_title(f"{peri}", va='bottom', fontsize=f_size, fontname=fontname)

cf =show(arr, ax=ax, transform=src.transform, cmap='PuOr_r', norm=colors.TwoSlopeNorm(vmin=range_min, vcenter=0., vmax=range_max)) #norm=colors.TwoSlopeNorm(0),  set zero as center
ax.coastlines(linewidth=.5)
ax.add_feature(cfeature.BORDERS, linewidth=.5, linestyle='-', edgecolor='black')
ax.set_xticks([])
ax.set_yticks([])

src.close()    
       
### set common color bar
cf_for_bar = cf.get_images()[0] #obtain image information
cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
plt.colorbar(cf_for_bar, ax=ax, cax=cbar_ax, orientation="horizontal", location='bottom',
             norm=colors.Normalize(vmin=range_min, vmax=range_max),
             )

plt.savefig(out_dir + os.sep + 'GOSIF_yearly_trend_by_MannKendall.png', dpi=600)



### old 
# gdf_boundary = gpd.read_file(national_boundary)

# fig,ax = plt.subplots(figsize=(8,8))

# raster_tempe = rasterio.open(tif_tempe)
# data_tempe = raster_tempe.read(1)

# base_ax1 = plt.imshow(data_tempe, cmap='Oranges') #いっかいこれを含めて回して後でコメントアウトするとできる気がする
# p1  = show(data_tempe, ax=ax, transform=raster_tempe.transform, cmap='Oranges', title='Slope_Temprature')

# # show(data_tempe, transform=raster_tempe.transform, ax=ax, cmap='Greys')
# cbar = fig.colorbar(base_ax1, ax=ax, shrink=0.4)
# cbar.set_label('Slope_Temprature')

# gdf_boundary.plot(ax=p1, color='none', edgecolor='black',linewidth=1) #ax=p1

# fig,ax = plt.subplots(figsize=(8,8))

# raster_gs = rasterio.open(tif_gsmap)
# data_gs = raster_gs.read(1)

# # base_ax1 = plt.imshow(data_gs, cmap='PuOr') #いっかいこれを含めて回して後でコメントアウトするとできる気がする
# p1  = show(data_gs, ax=ax, transform=raster_gs.transform, cmap='PuOr', title='Slope_Precipitation')

# cbar = fig.colorbar(base_ax1, ax=ax, shrink=0.4)
# cbar.set_label('Slope_Precipitation')

# gdf_boundary.plot(ax=p1, color='none', edgecolor='black',linewidth=1) #ax=p1



"""#Lines sample"""

centerline = r"D:\Malaysia\01_Brueprint\09_Terrace_detection\4_centerlines\centerlines.shp"
nonerror = r"D:\Malaysia\01_Brueprint\09_Terrace_detection\11_connect_lines\centerlines_45_cut_cut2ls_45_connectSQ2\merge\_centerlines_45_cut_cut2ls_merge_45_connect_sq_merge.shp"
paired = r"D:\Malaysia\01_Brueprint\12_Pairing_terraces\5_pairing\_post\_direction\merge\merge_cut_cut2_vertical_post_T1T2_post_dire.shp"
target_area = r"F:\MAlaysia\Blueprint\01_Boundary\target_sloping_area_diss.shp"
target_extent = r"D:\Malaysia\01_Brueprint\15_Paper\line_sample_extent.shp"

# gdf_target = gpd.read_file(target_area, crs='EPSG:32648')

# ulx = 225950 #226200
# uly = 268420 #268100
# extent = [ulx, ulx + 80, uly-80, uly ]#[xmin, xmax, ymin, ymax]
# extent_polygon = Polygon([(extent[0], extent[2]), (extent[1], extent[2]),
#                    (extent[1], extent[3]), (extent[0], extent[3])])
# gdf_extent = gpd.GeoDataFrame(geometry=[extent_polygon], crs='EPSG:32648')

## using shape file
gdf_extent = gpd.read_file(target_extent)

# target areaと見たいextentでクリップ
def clip_shp(shpfile):
   gdf = gpd.read_file(shpfile)
   # gdf_clipped_target = gpd.overlay(gdf, gdf_target, how="intersection")
   # gdf_clipped_extent = gpd.overlay(gdf_clipped_target, gdf_extent, how="intersection", keep_geom_type=False)
   gdf_clipped_extent = gpd.overlay(gdf, gdf_extent, how="intersection", keep_geom_type=False)

   return gdf_clipped_extent

gdf_center = clip_shp(centerline)
gdf_nonerror = clip_shp(nonerror)
gdf_paired = clip_shp(paired)

fig, axs = plt.subplots(2, 2, figsize=(10,6.5))
plt.subplots_adjust() #hspace=0.2, wspace=-0.6

fontsize = 18

idx = (0,0)
# gdf_center.plot(ax=axs[0])
gdf_center.plot(ax=axs[idx])
axs[idx].set_title(f'(a)', fontsize=fontsize)
start, end = axs[idx].get_xlim()
stepsize=50
axs[idx].xaxis.set_ticks(np.arange(start, end, stepsize))

idx = (0,1)
gdf_nonerror.plot(ax=axs[idx])
axs[idx].set_title(f'(b)', fontsize=fontsize)
axs[idx].set_yticks([])
axs[idx].xaxis.set_ticks(np.arange(start, end, stepsize))

idx = (1,0)
gdf_paired.plot(ax=axs[idx],column="T1T2", cmap= "brg")
axs[idx].set_title(f'(c)', fontsize=fontsize)
# axs[idx].set_yticks([])
axs[idx].xaxis.set_ticks(np.arange(start, end, stepsize))

idx = (1,1)
gdf_paired.plot(ax=axs[idx],column="Pair", cmap= "flag")
axs[idx].set_title(f'(d)', fontsize=fontsize)
axs[idx].set_yticks([])
axs[idx].xaxis.set_ticks(np.arange(start, end, stepsize))

# Adjust layout for better spacing
plt.tight_layout()

fig.savefig(r"D:\Malaysia\01_Brueprint\15_Paper\lines.png")

"""#CSimage sample"""

dem_tif = '/content/drive/MyDrive/Malaysia/Blueprint/CSMap/01_R_Out/DEM_05m_R_kring.tif'
slope_tif = '/content/drive/MyDrive/Malaysia/Blueprint/CSMap/01_R_Out/DEM_05m_R_kring_slope.tif'
curvature_tif = '/content/drive/MyDrive/Malaysia/Blueprint/CSMap/01_R_Out/DEM_05m_R_kring_curv.tif'
cs_tif = "/content/drive/MyDrive/Malaysia/Blueprint/CSMap/IMAGE/R_kring_filt_CS_05.tif"

src_dem = rasterio.open(dem_tif)
src_slope = rasterio.open(slope_tif)
src_curv = rasterio.open(curvature_tif)
src_cs = rasterio.open(cs_tif)

show(src_cs)

### rasterio windowの場合、row,  colのindex指定になる。座標からインデックスを取得する
#https://rasterio.readthedocs.io/en/stable/api/rasterio.transform.html

# custom_extent = [226230, 226430, 268200, 268400] #[xmin, xmax, ymin, ymax]
row, col = src_dem.index(226000, 268120) #UL
print(row, col) #これは合ってる
window = Window(col, row, 200, 200) #col_off, row_offの順
window

#test
# import geopandas as gpd
# from shapely.geometry import Point
# x, y = 226230, 268400

# # Create a GeoDataFrame with a single point
# geometry = [Point(x, y)]
# gdf = gpd.GeoDataFrame(geometry=geometry, crs='EPSG:32648')

# # Save the GeoDataFrame to a shapefile
# out_dir = "/content/drive/MyDrive/Malaysia/Blueprint/99_test"
# output_shapefile = out_dir + '/index_point_test.shp'
# gdf.to_file(output_shapefile)

# arr_dem =src_dem.read(1)
# arr_dem[row, col]
# arr_slope =src_slope.read(1)
# arr_slope[row, col]

dem_slice = src_dem.read(1,window=window)
slope_slice = src_slope.read(1,window=window)
curv_slice = src_curv.read(1,window=window)
cs_slice = src_cs.read(window=window)
cs_slice_rev = np.transpose(cs_slice, (1, 2, 0)) #(row, col, bands)のshapeに変更

# transform = src_dem.window_transform(window) #UL
# transform

fig, axs = plt.subplots(1, 4)

plt.subplots_adjust(wspace=-0.6) #hspace=0.2 #マイナス設定ができた

axs[0].imshow(dem_slice, cmap='Greys_r')
axs[0].set_title('DEM')

axs[1].imshow(slope_slice, cmap='Reds', interpolation='nearest')
axs[1].set_title('Slope')

axs[2].imshow(curv_slice, cmap='Blues', interpolation='nearest')
axs[2].set_title('Curvature')

axs[3].imshow(cs_slice_rev[:,:,0:3])
axs[3].set_title('CS image')

# Remove axis labels
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()

# plt.figure(figsize=(10,10))
fig, axs = plt.subplots(2, 2)

plt.subplots_adjust(wspace=-0.6) #hspace=0.2 #マイナス設定ができた

axs[0, 0].imshow(dem_slice, cmap='Greys_r')
axs[0, 0].set_title('DEM')

axs[0, 1].imshow(slope_slice, cmap='Reds', interpolation='nearest')
axs[0, 1].set_title('Slope')

axs[1, 0].imshow(curv_slice, cmap='Blues', interpolation='nearest')
axs[1, 0].set_title('Curvature')

axs[1, 1].imshow(cs_slice_rev[:,:,0:3])
axs[1, 1].set_title('CS image')

# Remove axis labels
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()
