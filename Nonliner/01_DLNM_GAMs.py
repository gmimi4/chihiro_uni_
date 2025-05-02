# -'- coding: utf-8 -'-
"""
distributed lag non-linear models (DLNM)
generalized additive models (GAMs)
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
from rpy2.robjects.packages import isinstalled
from rpy2.robjects import pandas2ri, r
import rpy2.robjects.packages as rpackages
from rpy2.robjects import Formula
from rpy2.robjects.vectors import StrVector


monthly_mean_dir = r"D:\Malaysia\02_Timeseries\CCM\01_region_mean\EVI"
out_dir = r"D:\Malaysia\02_Timeseries\YieldWater\12_mean_sd_plot"

""" # memo"""
# Prepare laged x data as well


# Activate automatic conversion
pandas2ri.activate()

# Load R packages
base = rpackages.importr('base')
utils = rpackages.importr('utils')
mgcv = rpackages.importr('mgcv')
dlnm = rpackages.importr('dlnm')
splines = rpackages.importr('splines')


# Example data
n = 100
df = pd.DataFrame({
    'y': np.random.normal(0, 1, n),
    'x': np.random.uniform(10, 30, n)
})

# Add lagged x values (simple example: lag 0 to 3)
for lag in range(4):
    df[f'x_lag{lag}'] = df['x'].shift(lag)
df.dropna(inplace=True)

# Send data to R
r_df = pandas2ri.py2rpy(df)

# Create crossbasis object in R
r.assign('data_r', r_df)
r('''
library(dlnm)
library(mgcv)

# Define crossbasis with splines and lags
cb <- crossbasis(data_r$x, lag=3, argvar=list(fun="bs", degree=2), arglag=list(fun="ns", df=3))

# Fit GAM with crossbasis term
model <- gam(y ~ cb, data=data_r)

# Store model summary
model_summary <- summary(model)
''')

# Get summary output
print(r('model_summary'))

    
        
    
    
    
    
    
    
    

    
    





           

