# -*- coding: utf-8 -*-
"""
# fourie transform SIF monthly original
# https://qiita.com/beasugerdad/items/a1efc187fcd5101ef0c2 #これがstraight forward
# Medium: https://medium.com/@khairulomar/deconstructing-time-series-using-fourier-transform-e52dd535a44e
# https://werk.ah.nl/blog/11/time-series-as-a-signal-fast-fourier-transform-to-decompose-seasonality
# Scipy: https://docs.scipy.org/doc/scipy/tutorial/fft.html
"""
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL #test
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy import fftpack


# csvfile = "/Volumes/PortableSSD/Malaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels/A1/15706.csv"
# csvfile=csv_file_list[15706]
csvfile = r"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels\A1\7443.csv" #7443 #15706
out_dir = r"D:\Malaysia\02_Timeseries\Resilience\02_fourier\_png"

startyear = 2002
endyear = 2022


def main(csvfile, startyear, endyear, time_lag):
    
    filename = os.path.basename(csvfile)[:-4]

    df_csv = pd.read_csv(csvfile, index_col = 'datetime',
                         parse_dates=['datetime'])
        
    """ # set period"""
    df_csv = df_csv[((df_csv.index.year >= startyear) & (df_csv.index.year <= endyear)) ]
    
    ### extraxt SIF original data
    df_csv_sif = df_csv.loc[:,"GOSIF"]          
    
    ### nanは外挿補完する
    df_sif_valid = df_csv_sif.interpolate() #nan線形補完
    
    
    """　#Extract trend and seasonality by moving average """
    stats_result = seasonal_decompose(df_sif_valid, period=12, two_sided=True) #
    stats_result.plot()

    ## comparison with STL
    stl_result = STL(df_sif_valid, period=12).fit()
    stl_result.plot()
    
    # stats_trend = stats_result.trend
    # stats_seasonal = stats_result.seasonal
    stats_trend = stl_result.trend
    stats_seasonal = stl_result.seasonal
    
    """# fourier transform for seasonality"""
    # サンプル数
    N = len(stats_seasonal)
    
    # フーリエ変換する
    y = stats_seasonal.values
    yf = fft(y)
    
    # 周波数軸の作成
    # サンプリング間隔
    # T = 1
    T = 1/12
    xf = fftfreq(N, T)[:N//2] #T: sampling space
    
    fig = plt.figure()
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    
    #save sample
    outfilename = out_dir + os.sep + f"ampli_freq_{filename}_stl.png"
    # outfilename = out_dir + os.sep + f"ampli_freq_{filename}_average.png"
    fig.savefig(outfilename)
    
    
    """# Inverse FFT from freq domain to original time domain"""
    inv = fftpack.ifft(yf)
    
    fig = plt.figure()
    plt.plot(stats_seasonal.index, inv)
    
    fig = plt.figure()
    plt.plot(stats_seasonal.index, stats_seasonal)
    

"""#Chart"""

# import matplotlib.pyplot as plt
# values_graph = [importance_dic[v] for v in vars]
# plt.barh(vars_rename, values_graph)
# title_str = os.path.basename(outfilename)[:-4]
# plt.tick_params(labelsize=18)

# plt.title(title_str)

# plt.savefig(out_dir+f"/{title_str}.png")

# values_graph

# vars

# """# Export txt"""

# #Export to txt
# df_importance = pd.DataFrame(importance_dic.values(),index=importance_dic.keys()).rename(columns={0:"importance"})
# df_importance

# df_importance.to_csv(outfilename)

# if __name__ == "__main__":
#     main()