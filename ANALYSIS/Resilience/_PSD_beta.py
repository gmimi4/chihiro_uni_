# -*- coding: utf-8 -*-
"""
# Wetland Dynamics Inferred from Spectral Analyses of Hydro-Meteorological Signals and Landsat Derived Vegetation Indices
# doi:10.3390/rs12010012
# calcurate PSD and slope
"""
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy import fftpack
from scipy import signal
import math
from scipy.optimize import curve_fit


""" test """
# startyear = 2002
# endyear = 2022

# csvfile = r"F:\MAlaysia\ANALYSIS\02_Timeseries\CPA_CPR\1_vars_at_pixels\A1\8581.csv" #7443

# df_csv = pd.read_csv(csvfile, index_col = 'datetime',
#                      parse_dates=['datetime'])

# filename = os.path.basename(csvfile)[:-4]

# """ # AMSREとAMSR2のギャップを補正する"""   
# pi_e = df_csv.loc[:,["SMDSCE", "VODDSCE"]]
# pi_2 = df_csv.loc[:,["SMDSC2", "VODDSC2"]]
# # calc median
# pi_e_med = pi_e.median()
# pi_2_med = pi_2.median()
# ratio_sm = pi_2_med.SMDSC2 /pi_e_med.SMDSCE
# ratio_vod = pi_2_med.VODDSC2 /pi_e_med.VODDSCE

# df_csv["SMDSCErev"] = df_csv["SMDSCE"] *ratio_sm
# df_csv["VODDSCErev"] = df_csv["VODDSCE"] *ratio_vod


# """ #SMとVODは平均にする"""    
# # try: #AMSREとAMSR2を一列にする
# df_csv["SM"] = df_csv[["SMDSCErev", "SMDSC2"]].mean(skipna=True, axis='columns')
# df_csv["VOD"] = df_csv[["VODDSCErev", "VODDSC2"]].mean(skipna=True, axis='columns')
# df_csv= df_csv.drop(["SMDSCE","SMDSC2","VODDSCE","VODDSC2", "SMDSCErev", "VODDSCErev"], axis=1)


# """ # set period"""
# df_csv = df_csv[((df_csv.index.year >= startyear) & (df_csv.index.year <= endyear)) ]

# """ # extract SIF"""
# df_csv_sif = df_csv.loc[:, "GOSIF"]

# """ # z scoring"""
# seri_csv_sif_z = df_csv_sif.copy()
# df_csv_sif_z = seri_csv_sif_z.to_frame()
# sif_mean = df_csv_sif_z["GOSIF"].mean()
# sif_std = df_csv_sif_z["GOSIF"].std()
# df_csv_sif_z.loc[:, "GOSIFz"] = (df_csv_sif_z["GOSIF"]-sif_mean)/sif_std

# data_sif = df_csv_sif_z["GOSIFz"]#.values #some GOSIF pixels not exist in early years

# ### divide period
# my_bkps_divi_rev =[[0, 84], [166, 251]]

# for ilst in my_bkps_divi_rev: #my_bkps_divi #[[0, 84], [166, 251]]
#     df_divi = data_sif[ilst[0]:ilst[1]+1]
#     # df_divi = data_sif
    
#     ### nanは外挿補完する
#     df_divi_valid = df_divi.interpolate() #nan線形補完
#     df_divi_valid = data_sif.interpolate() #nan線形補完
#     ## drop nan completely
#     df_divi_valid = df_divi_valid[~np.isnan(df_divi_valid)]
    
#     """　#Extract trend and seasonality """
#     ## stasmodel
#     stats_result = seasonal_decompose(df_divi_valid, period=12, two_sided=True) #
#     stats_trend = stats_result.trend ## statsmodel
#     stats_seasonal = stats_result.seasonal #it seems average after detrended
#     stats_seasonal_true = stats_result.observed - stats_trend # pure subtraction
    
#     ## STL
#     stl_result = STL(df_divi_valid, period=12).fit()
#     stl_seasonal = stl_result.seasonal
#     stl_seasonal_true = stl_result.observed - stl_result.trend
""" test ここまで """

def main(use_data):  ## series data
    
    """# fourier transform for seasonality"""
    ## set data
    # use_seasonal = stl_result.observed
    # stats_seasonal_vali = stats_seasonal[~np.isnan(stats_seasonal)]
    
    # サンプル数
    N = len(use_data)
    
    # フーリエ変換する
    y = use_data.values
    yf = fft(y) #yf = fft(xcoh) #yf = fft(ycoh)
    
    # 周波数軸の作成
    # サンプリング間隔
    # T = 1
    T = 1/12 #T: sampling space, inverse of the sampling rate #sampling rateとは1秒間に実行する処理回数
             #1年間に実行する回数とする？
    xf = fftfreq(N, T)[:N//2] #fftfreq returns the FFT sample frequency points.
    
    
    ### Calc by hand 出力されるｘ軸の順序がおかしい
    # http://www-mete.kugi.kyoto-u.ac.jp/zaki/db/lec1-3.html
    # パワースペクトル密度に変換(PSD)
    # pw = abs(yf*2/N)**2
    # dT = 1 # データ間隔-> 1 month
    # domg = 2.0*np.pi/(N*dT) #最小周波数
    # psd = pw/domg
    
    # # 片側スペクトルに変換
    # psd_oneside = psd[0:int(N/2)+1]*2 # 最初の半分をとってきて２倍する
    # psd_oneside[int(N/2)] /= 2.0 ## 0, N/2の成分(N:偶数のとき）は二倍しなくて良いので元に戻す（N/2の成分はほとんど寄与しないが）
    # psd_oneside[0] /= 2.0
    
    # #### 周波数軸の準備---------
    # # 周波数
    # xfft = np.linspace(0.0,domg*(N/2),int(N/2)+1)
    
    # # 周期（month）に変換
    # xfft_T = 2*(np.pi)/xfft #?ここあってる？->monthに変換
    
    """#　描画
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    # PSD
    ax.plot(1/xfft_T, psd_oneside,label='PSD') #1/monthで結局Hzになる
    ##（注意）a[start:end]としたとき、endは含まないことに注意
    
    # omega*PSD (x軸をlog座標で表示する際にはこうすると各周波数帯のエネルギーの寄与が見やすい)
    # ax.plot(xfft_T[1:int(N/2)+1],psd_oneside[1:int(N/2)+1]*xfft[1:int(N/2)+1],label='ω*PSD')
    
    #設定
    ax.set_yscale('log') # 軸を対数表示に設定
    ax.set_xscale('log')
    # ax.invert_xaxis() # X軸を反転
    ax.set_title('Power Spectrum',size='x-large') # タイトルとか
    ax.legend(loc=2) # 凡例
    plt.show()
    """
    
    ### Calc by Scipy
    fs = 1 #fs = domg
    f, Pxx_den = signal.welch(use_data.values, fs, nperseg=None)
    
    # #plot f
    # plt.semilogy(f, Pxx_den)
    # # plt.ylim([0.5e-3, 1])
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('PSD [V**2/Hz]')
    # plt.show()
    
    """
    #plot power low
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    # PSD
    # ax.plot(f, Pxx_den, label='PSD') #これ
    ax.plot(f, Pxx_den, label='PSD')
    #設定
    ax.set_yscale('log') # 軸を対数表示に設定
    ax.set_xscale('log')
    # ax.invert_xaxis() # X軸を反転? -> no need?
    ax.set_title('Power Spectrum',size='x-large') # タイトルとか
    ax.legend(loc=2) # 凡例
    plt.show()
    """
    
    ### Fitting
    ## eliminate f=0
    f = f[1:]
    Pxx_den = Pxx_den[1:]
    # find inf
    inf_indices_f = np.where(np.isinf(f))[0]
    inf_indices_y = np.where(np.isinf(Pxx_den))[0]
    # concat
    inf_indices = np.concatenate([inf_indices_f, inf_indices_y])
    # clean
    f_rev = np.delete(f, inf_indices)
    y_rev = np.delete(Pxx_den, inf_indices)
    
    #This is mistake...→これでいい
    logf = np.log10(f)
    logy = np.log10(Pxx_den)
    
    inf_indices_f = np.where(np.isinf(logf))[0]
    inf_indices_y = np.where(np.isinf(logy))[0]
    # concat
    inf_indices = np.concatenate([inf_indices_f, inf_indices_y])
    # clean
    logf_rev = np.delete(logf, inf_indices)
    logy_rev = np.delete(logy, inf_indices)
    
    liner_fitting = np.polyfit(logf_rev, logy_rev, 1)
    #array([a, b])
    
    return liner_fitting[0]
    
    
    # """ 累乗近似 """ #
    # def exp_func(x, a, b): #degine func
    #     return b*(x**a)
    # def exp_fit(val1_quan, val2_quan):
    #     # maxfev：関数の呼び出しの最大数, check_finite：Trueの場合NaNが含まれている場合はValueError発生
    #     l_popt, l_pcov = curve_fit(exp_func, val1_quan, val2_quan, check_finite=False)
    #     # return exp_func(val1_quan, *l_popt),l_popt
    #     return l_popt
    
    # ##Fitting
    # l_popt = exp_fit(f_rev, y_rev)
    
    
    # return l_popt[0]
    
    """
    fig = plt.figure()
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2])) ## ここで2/Nやってた

    out_dir_fin = out_dir + os.sep + "_png"
    outfilename = out_dir_fin + os.sep + f"ampli_freq_{filename}_{ilst[0]}-{ilst[1]}.png"
    # outfilename = out_dir_fin + os.sep + f"ampli_freq_{filename}_{ilst[0]}-{ilst[1]}_ADFdiff.png"
    fig.savefig(outfilename)
    """