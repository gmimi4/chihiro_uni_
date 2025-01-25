# -*- coding: utf-8 -*-
"""
Cartopy is in gdal_copy (spyder ok) and rasterio_copy2 (no spyder)
"""

import matplotlib.pyplot as plt
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
plt.rcParams['font.family'] = 'Times New Roman'

# -------------------------------------
"""# Yield trend negative to positive"""
# -------------------------------------

import matplotlib.pyplot as plt
import os
import geopandas as gpd
import matplotlib.colors as mcolors

yeild_shp = r"D:\Malaysia\Validation\1_Yield_doc\shp\region_slope_fin.shp"
filename = os.path.basename(yeild_shp)[:-4]
out_dir = os.path.dirname(yeild_shp) + os.sep + "_png"

gdf_region = gpd.read_file(yeild_shp)
gdf_region_filer = gdf_region[gdf_region.slope!=0]

norm = mcolors.TwoSlopeNorm(vmin=gdf_region_filer['slope'].min(), vmax=gdf_region_filer['slope'].max(), vcenter=0)
# norm = mcolors.TwoSlopeNorm(vmin=-1, vmax=1, vcenter=0)

fontname='Times New Roman'
f_size_title = 20
f_size = 16            
fig, ax = plt.subplots(1,1, figsize=(10,5))
plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
fig.tight_layout()
# plot colored
gdf_region_filer.plot(column='slope', cmap='coolwarm', legend=True, ax=ax, edgecolor='black', 
                      linewidth=0.5, norm=norm, legend_kwds={'shrink': 0.7})
# plot boarder
gdf_region.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
# ax.set_xticks([])
# ax.set_yticks([])

plt.savefig(out_dir + os.sep + f'{filename}_slope.png', dpi=600)
plt.close()

    

# -------------------------------------
"""# Rank plot"""
# -------------------------------------

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import os
import geopandas as gpd
import numpy as np
import glob

shp_dir = r"D:\Malaysia\02_Timeseries\Resilience\05_rank_correlation\_halfperiod\_rank_shp"
# shp_dir = r"D:\Malaysia\02_Timeseries\Resilience\05_rank_correlation\_pelt\_rank_shp"
shps = glob.glob(shp_dir + os.sep + "*.shp")

def plot_rankshp(shp, rankfield):
    
    filename = os.path.basename(shp)[:-4]
    out_dir = os.path.dirname(shp) + os.sep + "_png"
    # out_dir = r"D:\Malaysia\02_Timeseries\Resilience\05_rank_correlation" 
    gdf_region = gpd.read_file(shp)
    # Filter out rows where value is 0(nan)
    gdf_filtered = gdf_region[gdf_region[rankfield].apply(lambda x: x>0)]
    
    # # ### rank coloring setting ###
    # if "psd" in filename: #negative rank correlation with yield
    #     clmap = 'cividis'
    # else:
    #     clmap = 'cividis_r' #opposite to yield
    
    clmap = 'cividis_r' ## all same as yield
    
    fontname='Times New Roman'
    f_size_title = 20
    f_size = 16            
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
    fig.tight_layout()
    # plot colored
    gdf_filtered.plot(column=rankfield, ax=ax, cmap=clmap, linewidth=0, legend=True, legend_kwds={'shrink': 0.7}) #legend=True, legend_kwds={'shrink': 0.3}
    # plot boarder
    gdf_region.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
    # ax.set_xticks([])
    # ax.set_yticks([])
    
    plt.savefig(out_dir + os.sep + f'{filename}.png', dpi=600)
    plt.close()


for shp in shps:
    plot_rankshp(shp, 'rank_tif')

# -------------------------------------
"""# Categorical plot"""
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

tif = r"D:\Malaysia\02_Timeseries\Resilience\03_change_point\_mosaic\mosaic_NumChangePoints_PELT.tif"

out_dir = os.path.dirname(tif) + os.sep + "_png"

extent_lonlat = (92, 147, -12, 9)
crs_lonlat = ccrs.PlateCarree()

fontname='Times New Roman'
f_size_title = 20
f_size = 16

categories =  {0: '0', 1: '1', 2: '2',  3: '3'}
cmap = ListedColormap(['#dcdcdc','goldenrod', 'orangered', 'goldenrod']) #191970:darkblue
        
peri = os.path.basename(tif)[:-4].split("_")[-1]
fig, ax = plt.subplots(figsize=(10,5), subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
fig.tight_layout()
   
src = rasterio.open(tif)
arr = src.read(1)

title_str = "Number of change points"
fig.suptitle(title_str, fontsize=f_size_title, fontname=fontname)
ax.set_extent(extent_lonlat, crs=crs_lonlat)
# ax.set_title(peri, va='bottom', fontsize=f_size, fontname=fontname)

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
"""# Raster plot in map for one period """
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

def raster_plot(tif_sensitivity, titlestr):
    # tif_sensitivity = r"D:\Malaysia\02_Timeseries\Resilience\03_change_point\_mosaic\mosaic_amplitudeDiff_PELT.tif"
    out_dir = os.path.dirname(tif_sensitivity) + os.sep + "_png"
    #
    ## not used here
    # def raster_range(tif):
    #     with rasterio.open(tif) as src:
    #         arr = src.read(1)
    #     minval = np.nanmin(arr)
    #     maxval = np.nanmax(arr)
    #     return minval, maxval
    extent_lonlat = (92, 147, -12, 9)
    crs_lonlat = ccrs.PlateCarree()
    #
    fontname='Times New Roman'
    f_size_title = 20
    f_size = 16
    #        
    fig, ax = plt.subplots(figsize=(10,5), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
    fig.tight_layout()
    #   
    src = rasterio.open(tif_sensitivity)
    arr = src.read(1)
    #
    ## determin min and max of range
    # tif_range = raster_range(tif_sensitivity)
    arr_nan = arr[~np.isnan(arr)]
    range_min = np.percentile(arr_nan, 2)
    range_max = np.percentile(arr_nan, 98)
    # range_min = 0.0 #0.5
    # range_max = 2 #1.8
    #    
    fig.suptitle(titlestr, fontsize=f_size_title, fontname=fontname)
    ax.set_extent(extent_lonlat, crs=crs_lonlat)
    # ax.set_title(period_str, va='bottom', fontsize=f_size, fontname=fontname)
    #
    # cf = show(src, ax=ax, transform=src.transform, cmap='viridis') #norm=colors.TwoSlopeNorm(0),  set zero as center
    norm = colors.Normalize(vmin=range_min, vmax=range_max)
    cf = show(arr, ax=ax, transform=src.transform, cmap='viridis', norm=norm) #viridis #Blues
    ax.coastlines(linewidth=.5)
    ax.add_feature(cfeature.BORDERS, linewidth=.5, linestyle='-', edgecolor='black')
    ax.set_xticks([])
    ax.set_yticks([])
    #
    src.close()    
    #  
    ### set common color bar
    base_ax = cf.get_images()[0]
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
    cbar = fig.colorbar(base_ax, ax=ax, cax=cbar_ax, orientation="horizontal", location='bottom',
                 norm=colors.Normalize(vmin=range_min, vmax=range_max),
                 )
    # cbar.ax.tick_params(labelsize=6)
    #
    filename = os.path.basename(tif_sensitivity)[:-4]
    plt.savefig(out_dir + os.sep + f'{filename}.png', dpi=600)

## Execute
tifs = glob.glob(r"D:\Malaysia\02_Timeseries\Resilience\03_halfperiod\_mosaic" + os.sep + "*.tif")
# tifs = glob.glob(r"D:\Malaysia\02_Timeseries\Resilience\03_change_point\_mosaic" + os.sep + "*.tif")
tifs = [t for t in tifs if "all_" in t]
for tif in tifs:
    filename = os.path.basename(tif)[:-4]
    raster_plot(tif, filename)
    
# -------------------------------------
"""# ** change centering 0"""
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

in_dir = r"D:\Malaysia\02_Timeseries\Resilience\03_change_point\_mosaic"
# in_dir = r"D:\Malaysia\02_Timeseries\Resilience\03_halfperiod\_mosaic"
out_dir = in_dir + os.sep + "_png"

tifs = glob.glob(in_dir + os.sep + "*.tif")
tifs = [t for t in tifs if "Diff" in t]

# """ # obtain data range """
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

for vartif in tifs:
    #
    tifname = os.path.basename(vartif)[:-4]
    # if "amplitude" in tifname:
    #     title_str = "Amplitude change"
    # if "std" in tifname:
    #     title_str = "Std change"
    # if "trend" in tifname:
    #     title_str = "Trend change"
    #        
    src = rasterio.open(vartif)
    arr = src.read(1)
    #
    ## determin min and max of range
    arr_nan = arr[~np.isnan(arr)]
    range_min = np.percentile(arr_nan, 2)
    range_max = np.percentile(arr_nan, 98)
    # range_min, range_max = raster_range(vartif)
    #
    fig, ax = plt.subplots(1, 1, figsize=(10,5), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(wspace=0.1, hspace=0.1) #hspace=0.2 #マイナス設定ができた
    fig.tight_layout()
    #
    ax.set_extent(extent_lonlat, crs=crs_lonlat)
    ax.set_title(tifname, va='bottom', fontsize=f_size, fontname=fontname)
    #
    cf =show(arr, ax=ax, transform=src.transform, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=range_min, vcenter=0., vmax=range_max)) #norm=colors.TwoSlopeNorm(0),  set zero as center
    ax.coastlines(linewidth=.5)
    ax.add_feature(cfeature.BORDERS, linewidth=.5, linestyle='-', edgecolor='black')
    ax.set_xticks([])
    ax.set_yticks([])
    #
    src.close()    
    #     
    ### set common color bar
    cf_for_bar = cf.get_images()[0] #obtain image information
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02]) # [ left, bottom, width, height ]
    plt.colorbar(cf_for_bar, ax=ax, cax=cbar_ax, orientation="horizontal", location='bottom',
                 norm=colors.Normalize(vmin=range_min, vmax=range_max),
                 )
    #
    outfilename = tifname +".png"
    plt.savefig(out_dir + os.sep + outfilename, dpi=600)

