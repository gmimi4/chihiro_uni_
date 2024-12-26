# -*- coding: utf-8 -*-
"""
Make 0.1 degree grid from sample tif
SIF／EVI以外のラスターでインデックスがずれていそうなため確認のため作成する
"""
# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install factor_analyzer

import numpy as np
import pandas as pd
import os,sys
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('/Volumes/SSD_2/Malaysia/02_Timeseries/CPA_CPR/0_vars_timeseries/EVI/A4/A4_VPD_pixels_dates.csv')
# test = df.iloc[:,0:3]
test = df.loc[:,"2000-01-01"]
print(test)
