# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 17:15:57 2024

@author: chihiro
"""

from statsmodels.tsa.stattools import adfuller
import pandas as pd
import pymannkendall as mk
import numpy as np
import pandas as pd

def ADF(df_unseasonal):
    dftest = adfuller(df_unseasonal, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "Lags Used",
            "Number of Observations Used",
        ],)
    
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value #Series
    
    pval_adf = dfoutput["p-value"]
    if pval_adf > 0.1: #null hypothesis is non-stationary
        slp_adf,intercept = np.polyfit(np.arange(len(df_unseasonal)), df_unseasonal.values, 1) #normal slope
    else:
        slp_adf = 0 #no trend
    
    return slp_adf


def MK(df_unseasonal, p_val):
    result = mk.original_test(df_unseasonal) # Yearly no seasonal data
    if result.p < p_val:
        slp = result.slope
    else:
        slp = 0
    
    return slp