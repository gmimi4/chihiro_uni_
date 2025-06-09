# -*- coding: utf-8 -*-
"""
# convert CORDEX nc file to tif
"""

import xarray as xr
import rasterio
from rasterio.transform import from_origin
import numpy as np
import os, sys
from rasterio.crs import CRS
crs = CRS.from_string("+proj=longlat +datum=WGS84 +no_defs")  # EPSG:4326

nc_path = r"E:\Malaysia\GCM\CORDEX\00_download\wgetfiles\Evaporation\CNRM\evspsbl_SEA-22_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_SMHI-RCA4_v1_mon_195101-196012.nc"
variable = sys.argv[1]#"evspsbl"

filename = os.path.basename(nc_path)[:-4]
evaluation = filename.split("_")[3]
dmodel = os.path.basename(os.path.dirname(nc_path))

output_dir = rf"E:\Malaysia\GCM\CORDEX\01_tif" + os.sep + variable + os.sep + dmodel
os.makedirs(output_dir, exist_ok=True)

ds = xr.open_dataset(nc_path)

data = ds[variable]  # shape: (time, lat, lon)
lats = ds['lat'].values
lons = ds['lon'].values

# Determine raster transform
res_x = np.abs(lons[0][1] - lons[0][0])
res_y = np.abs(lats[-1][0] - lats[-2][0])
transform = from_origin(west=lons.min(), north=lats.max(), xsize=res_x, ysize=res_y)

# Ensure lat is decreasing (north to south)
check = lats[0] < lats[-1]
check = np.sum(check)
if check>0:
    data = data[:, ::-1, :]  # flip lat axis

# --- Save each time step as GeoTIFF ---
for i in range(data.shape[0]):
    date_str = str(ds['time'].dt.strftime('%Y%m').values[i])  # e.g., '207001'
    out_tif = os.path.join(output_dir, f"{variable}_{dmodel}_{evaluation}_{date_str}.tif")
    array = data[i].values.astype(np.float32)

    with rasterio.open(out_tif,'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs="EPSG:4326",  # Most CORDEX data is in lat/lon
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(array, 1)