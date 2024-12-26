# -*- coding: utf-8 -*-
"""
numpy: https://labo-code.com/python/wavelet-coherence-pahse/
Using this: Pywavelet: https://atatat.hatenablog.com/entry/data_proc_python10
PyWavelt: https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
Scipy: https://qiita.com/yukiB/items/59f8484e72bb0471ad47
"""

import pandas as pd
import os, sys
import glob
import time
# import pywt
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

csv_dir = r"D:\Malaysia\02_Timeseries\CCM\01_region_mean"
out_dir = r"D:\Malaysia\02_Timeseries\Wavelet\01_cross_correlation"

startyear = 2002
endyear = 2023

csvs = glob.glob(csv_dir + os.sep + "*.csv")

varlist = ['GOSIF','rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD']
varlist2 = ['rain', 'temp', 'VPD', 'Et', 'Eb', 'SM', 'VOD'] 

# 連続ウェーブレット変換を行う関数
def CWT_Morlet(tm, sig): #tm, sig: series
    dj = 0.125
    dt = tm[1] - tm[0]
    n_cwt = len(sig) #int(2**(np.ceil(np.log2(len(sig))))) #512?

    # --- 後で使うパラメータを定義
    omega0 = 6.0 #Central frequency of the Morlet wavelet (ChatGPT)
    s0 = 2.0*dt #最小スケール2ヶ月（たぶん）
    J = int(np.log2(n_cwt*dt/s0)/dj) #64
    
    # --- スケール
    s = s0*2.0**(dj*np.arange(0, J+1, 1)) ## Practical guideよりs=s0*2.0**(J*dj)で最大スケール

    # --- n_cwt個のデータになるようにゼロパディングをして，DC成分を除く #オフセットぽい
    x = np.zeros(n_cwt)
    x[0:len(sig)] = sig[0:len(sig)] - np.mean(sig)

    # --- omega array
    omega = 2.0*np.pi*np.fft.fftfreq(n_cwt, dt) #rad/sec ω = 2pi/T = 2pif

    # --- FFTを使って離散ウェーブレット変換する
    X = np.fft.fft(x) # eq.(3) #離散フーリエ変換?
    cwt = np.zeros((J+1, n_cwt), dtype=complex) # CWT array #スケールと同順に値が入る（そりゃそう）

    Hev = np.array(omega > 0.0)
    for j in range(J+1):
        # Torrence 1998 eq.(6) and (1) #あってる？(1): Morlet
        # https://labo-code.com/python/wavelet/にスケールsのMorletの周波数空間表現の説明あり (Torrence1998 Table1)
        Psi = np.sqrt(2.0*np.pi*s[j]/dt)*np.pi**(-0.25)*np.exp(-(s[j]*omega-omega0)**2/2.0)*Hev
        cwt[j, :] = np.fft.ifft(X*np.conjugate(Psi)) #np.fft.ifft:逆フーリエ変換
        # By the convolution theorem, the wavelet transform is the inverse Fourier transform of the product
        # eq.4

    s_to_f = (omega0 + np.sqrt(2 + omega0**2)) / (4.0*np.pi)
    freq_cwt = s_to_f / s #Table1 Fourie wavelengthの逆数
    cwt = cwt[:, 0:len(sig)]
    # (practical guide)
    # one should certainly convert from scale to Fourier period before plotting, 
    # as presumably one is interested in equating wavelet power at a certain time
    # and scale with a (possibly shortlived) Fourier mode at the equivalent Fourier period

    # --- cone of interference
    COI = np.zeros_like(tm)
    COI[0] = 0.5/dt
    COI[1:len(tm)//2] = np.sqrt(2)*s_to_f/tm[1:len(tm)//2]
    COI[len(tm)//2:-1] = np.sqrt(2)*s_to_f/(tm[-1]-tm[len(tm)//2:-1])
    COI[-1] = 0.5/dt

    return s, cwt, freq_cwt, COI, J

def smoothing(dt, s, W): #dt: kernel widthみたいなもの
    dj = 3/12 #0.125 #boxcar filtering width #Torrence 1999
    n_s, n_t = W.shape
    W_smthd = W

    # --- 時間方向の平滑化
    #Torrence 1999 Appendix
    for j in range(n_s): #freq domain
        W_temp = np.concatenate([np.zeros_like(W[j, :]), W[j, :],np.zeros_like(W[j, :])]) #あるs（j）の全期間取得
        tm_wndw = np.arange(-np.round(n_t), np.round(n_t))*dt
        krnl = np.exp(-(tm_wndw)**2/(2*s[j]**2)) #フィルター
        W_smthd[j, :] = (np.convolve(W_temp, krnl, mode='same')/np.sum(krnl))[n_t:2*n_t]
        # np.convolve:移動平均をとる or 畳み込み、つまりウィンドウ内（kernel存在内）で積和
        # filter given by the absolute value of the wavelet function at each scale,
        # normalized to have a total weight of unity.

    # --- スケール方向の平滑化
    #Torrence 1999 Appendix
    # scale smoothing is done using a boxcar filter of width dj0
    for i in range(n_t):
        dcrr = int(np.floor(0.6/(2*dj)))
        krnl = np.concatenate([[np.mod(dcrr, 1)], np.ones(dcrr), [np.mod(dcrr, 1)]])
        W_smthd[:, i] = np.convolve(W_smthd[:, i], krnl, mode='same')/np.sum(krnl)

    return W_smthd



""" # Run """
for csvfile in tqdm(csvs):
    reginame = os.path.basename(csvfile)[:-4].split("_")[0]
    df_csv = pd.read_csv(csvfile, index_col = 'datetime',
                         parse_dates=['datetime'])
    
    ## set period
    df_csv = df_csv[((df_csv.index.year >= startyear) & (df_csv.index.year <= endyear)) ]
    
    ## Scaling 0-1""" ##
    for var in varlist:
        df_var = df_csv.loc[:,var]
        df_var_nan = df_var.dropna()
        var_max = df_var.max()
        var_min = df_var.min()
        df_csv.loc[:, f"{var}s"] = (df_var-var_min)/(var_max - var_min)
        num_year = len(set(list(df_csv.index.year)))
    
    #------------------------------------------------------------------------------
    for var in varlist2:
        sig01 = df_csv["GOSIFs"]
        sig02 = df_csv[f"{var}s"]
        
        sigcon = pd.concat([sig01, sig02], axis=1)
        sigcon = sigcon.interpolate(method='linear', axis=0)
        sigcon = sigcon.dropna()
        datetime = sigcon.index.to_numpy()
        
        sig01 = sigcon.loc[:,"GOSIFs"]
        sig02 = sigcon.loc[:,f"{var}s"]
        tm01 = np.arange(0, len(sigcon), 1)
        tm02 = tm01
        
        # 連続ウェーブレット変換を実行
        s01, cwt01, freq_cwt01, COI01, J01 = CWT_Morlet(tm01, sig01)
        s02, cwt02, freq_cwt02, COI02, J02 = CWT_Morlet(tm02, sig02)
                
        # クロススペクトル
        xwt = cwt01*np.conjugate(cwt02)   
        
        # コヒーレンスとフェイズ
        dt = 1 #1/(len(sig01)/12) #ガウシアンカーネルの時間値を*倍にして使うことになると思う
        cwt01_pw_smthd = smoothing(dt, s01, np.abs(cwt01)**2*np.array([1.0/s01]).T)
        cwt02_pw_smthd = smoothing(dt, s02, np.abs(cwt02)**2*np.array([1.0/s02]).T)
        xwt_smthd = smoothing(dt, s01, xwt*np.array([1.0/s01]).T)
        
        wcoh = np.abs(xwt_smthd)**2 / (cwt01_pw_smthd*cwt02_pw_smthd)
        wphs = np.rad2deg(np.arctan2(np.imag(xwt_smthd), np.real(xwt_smthd)))
            
        # -------------------------------------------------------
        ## Export resutlts
        out_dir_fin = out_dir + os.sep + reginame
        os.makedirs(out_dir_fin, exist_ok=True)
        np.save(out_dir_fin+os.sep + f"coherence_GOSIF_{var}.npy", wcoh)
        np.save(out_dir_fin+os.sep + f"phase_GOSIF_{var}.npy", wphs)
        np.save(out_dir_fin+os.sep + f"frequency_{var}.npy", freq_cwt01)
        np.save(out_dir_fin+os.sep + f"scale_{var}.npy", s01)
        np.save(out_dir_fin+os.sep + f"datetime_{var}.npy", datetime)
        
        # -------------------------------------------------------
        ### Plot Scarogram 
        cmap = mpl.cm.jet#cividis #gnuplot2 #mpl.cm.jet
        def plot_scaro(sig, freq_cwt, cwt, COI, varname):
            # freq_cwt = freq_cwt[::-1] #when plotting in scale #->以降ずっと順番が変わるのでやめる
            fig = plt.figure(figsize = (8,5))
            plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.15)
            ax = fig.add_subplot(2,1,1)
            # 元データのプロット
            ax.plot(tm01, sig, c='black', linewidth=0.5)
            ax.set_xticks([]) 
            ax.set_ylabel(f'{varname}(scaled)')
            # スカログラムのプロット
            ax = fig.add_subplot(2,1,2)
            # カラーバーのレンジの設定
            freq_lower = freq_cwt.min()
            freq_upper = freq_cwt.max()
            # log10の値にした方が小さなパワーのも拾えてぽく見える
            # norm = mpl.colors.Normalize(vmin=np.log10(np.abs(cwt[freq_cwt < freq_upper, :])**2).min(),
            #                             vmax=np.log10(np.abs(cwt[freq_cwt < freq_upper, :])**2).max())
            norm = mpl.colors.Normalize(vmin=(np.abs(cwt01[freq_cwt < freq_upper, :])**2).min(),
                                        vmax=(np.abs(cwt01[freq_cwt < freq_upper, :])**2).max())
            # ax.contourf(tm01, freq_cwt, np.log10(np.abs(cwt)**2),  #x,y,z #この順の格子とするぽい
            #                   norm=norm, levels=256, cmap=cmap)
            ax.contourf(tm01, freq_cwt, np.abs(cwt)**2,  #x,y,z #この順の格子とするぽい
                              norm=norm, levels=256, cmap=cmap)
            ### Cone of Interference のプロット
            ax.fill_between(tm01, COI, fc='w', hatch='x', alpha=0.5)
            ax.plot(tm01, COI, '--', color='black', linewidth=0.5)
            ax.set_ylim(freq_lower, freq_upper)
            ax.set_ylabel("frequency")
            ## to date
            jan_dates = pd.date_range(start=f"{startyear}", end=f"{endyear}", freq="AS")
            tick_positions = np.linspace(0, len(sig) - 1, len(jan_dates))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(jan_dates.strftime('%Y'), rotation=45, ha="right") #'%Y-%m-%d'
            # Colorbar
            cx_sc = fig.add_axes([0.91, 0.18, 0.02, 0.33]) #left, bottom, width, height
            mpl.colorbar.ColorbarBase(cx_sc, cmap=cmap, norm=norm,
                                      orientation="vertical",
                                      label='log10 (Wavelet Power)') #label='$\log_{10}|W^x|^2$'
            # plt.tight_layout()
            fig.savefig(out_dir_fin + os.sep + f"scarogram_{varname}.png", bbox_inches='tight')
            plt.close()
            
            
        # plot
        plot_scaro(sig01, freq_cwt01, cwt01, COI01, "GOSIF") #freq_cwt01
        plot_scaro(sig02, freq_cwt02, cwt02, COI02, var)
        
        
        # -------------------------------------------------------
        # テスト信号 sig01 と sig02 のウェーブレットコヒーレンスのプロット
        fig = plt.figure(figsize = (10,5))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.15)
        ax = fig.add_subplot(2,1,1)
        cmap = mpl.cm.jet
        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        ax.contourf(tm01, freq_cwt01, wcoh, norm=norm, cmap=cmap, levels=256) #levels=10, 
        freq_lower = freq_cwt01.min()
        freq_upper = freq_cwt01.max()
        ax.set_ylim(freq_lower, freq_upper)
        ax.set_ylabel("frequency")
        ax.set_xticks([])
        # ax.text(0.99, 0.97, "coherence", color='white', ha='right', va='top',
        #              path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
        #                            path_effects.Normal()], 
        #              transform=ax.transAxes)
        # Cone of Interference のプロット
        ax.fill_between(tm01, COI01, fc='w', hatch='x', alpha=0.5)
        ax.plot(tm01, COI01, '--', color='black')
        # Colorbar
        cx_sc = fig.add_axes([0.91, 0.6, 0.02, 0.33]) #left, bottom, width, height
        mpl.colorbar.ColorbarBase(cx_sc, cmap=cmap, norm=norm,
                                  orientation="vertical",
                                  label='coherence')
        # ---------------------------
        # テスト信号 sig01 と sig02 のウェーブレットフェイズのプロット
        ax = fig.add_subplot(2,1,2)
        cmap = mpl.cm.twilight#hsv
        norm = mpl.colors.Normalize(vmin=-180.0, vmax=180.0)
        # ax.contourf(tm01, freq_cwt01, np.where(wcoh >= 0.75, wphs, np.nan), #コヒーレンスが0.75以上の場合にのみ表示
        ax.contourf(tm01, freq_cwt01, wphs, norm=norm, cmap=cmap, levels=256) #levels=16 
        # ax.text(0.99, 0.97, "phase", color='white', ha='right', va='top',
        #              path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
        #                            path_effects.Normal()], 
        #              transform=ax.transAxes)
        ax.set_ylim(freq_lower, freq_upper)
        ax.set_ylabel("frequency")
        # Cone of Interference のプロット
        ax.fill_between(tm01, COI01, fc='w', hatch='x', alpha=0.5)
        ax.plot(tm01, COI01, '--', color='black')
        # Colorbar
        cx_sc = fig.add_axes([0.91, 0.15, 0.02, 0.33]) #left, bottom, width, height
        mpl.colorbar.ColorbarBase(cx_sc, cmap=cmap, norm=norm,
                                  orientation="vertical",
                                  label='phase')
        out_dir_fin = out_dir + os.sep + reginame
        fig.savefig(out_dir_fin + os.sep + f"coherence_phase_GOSIG_{var}.png", bbox_inches='tight')
        plt.close()

