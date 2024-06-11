# -*- coding: utf-8 -*-
"""
#https://www.kaggle.com/code/gianinamariapetrascu/pca-varimax-rotation
# https://www.youtube.com/watch?v=BiuwDI_BbWw
"""
# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install factor_analyzer

import numpy as np
import pandas as pd
import os
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from sklearn.linear_model import Ridge
import statsmodels.api as sm
# from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler


# csvfile = "/Volumes/PortableSSD 1/MAlaysia/ANALYSIS/02_Timeseries/CPA_CPR/1_vars_at_pixels/A1/15706.csv"
# csvfile = r"D:\Malaysia\02_Timeseries\CPA_CPR\1_vars_at_pixels\A1\0.csv"
# startyear = 2002
# endyear = 2012

def main(csvfile, pval, startyear, endyear):
    # print(csvfile)
    
    # df_csv = pd.read_csv(csvfile, index_col = 'datetime',
    #                      date_parser=lambda x:pd.to_datetime(x,format ="%Y-%m-%d")) #future warning
    df_csv = pd.read_csv(csvfile, index_col = 'datetime',
                         parse_dates=['datetime'])
    

    """ # AMSREとAMSR2のギャップを補正する"""
    # ratio_sm = '/Volumes/PortableSSD 1/MAlaysia/ANALYSIS/03_AMSR_comparison/SM_ratio.txt'
    # ratio_vod = '/Volumes/PortableSSD 1/MAlaysia/ANALYSIS/03_AMSR_comparison/VOD_ratio.txt'    
    # df_ratio_sm = pd.read_csv(ratio_sm)
    # df_ratio_vod = pd.read_csv(ratio_vod)
    # ratio_sm = df_ratio_sm.ratio.values[0]
    # ratio_vod = df_ratio_vod.ratio.values[0]
    
    pi_e = df_csv[((df_csv.index.year < 2012) & (df_csv.index.month < 11))]
    pi_2 = df_csv[(df_csv.index.year >= 2012) & (df_csv.index.year < 2023)]
    # calc median
    pi_e_med = pi_e.median()
    pi_2_med = pi_2.median()
    ratio_sm = pi_2_med.SMDSC2 /pi_e_med.SMDSCE
    ratio_vod = pi_2_med.VODDSC2 /pi_e_med.VODDSCE
    
    df_csv["SMDSCErev"] = df_csv["SMDSCE"] *ratio_sm
    df_csv["VODDSCErev"] = df_csv["VODDSCE"] *ratio_vod
    
    
    """ #SMとVODは平均にする"""
    #skipnaでいけた
    # try: #columnにDSCとASCがある場合
    #     df_csv["SM"] = df_csv[["SMASC", "SMDSC"]].mean(skipna=True, axis='columns')
    #     df_csv["VOD"] = df_csv[["VODASC", "VODDSC"]].mean(skipna=True, axis='columns')
    #     df_csv= df_csv.drop(["SMASC","SMDSC","VODASC","VODDSC"], axis=1)
    
    # try: #AMSREとAMSR2を一列にする
    df_csv["SM"] = df_csv[["SMDSCErev", "SMDSC2"]].mean(skipna=True, axis='columns')
    df_csv["VOD"] = df_csv[["VODDSCErev", "VODDSC2"]].mean(skipna=True, axis='columns')
    df_csv= df_csv.drop(["SMDSCE","SMDSC2","VODDSCE","VODDSC2", "SMDSCErev", "VODDSCErev"], axis=1)
    # except: #片方だけの場合
    #     vars_list = df_csv.columns.tolist()
    #     sv_col_list = [c for c in vars_list if "SM" in c or "VOD" in c]
    #     for colname in sv_col_list:
    #         if "SM" in colname:
    #             newname = "SM"
    #         elif "VOD" in colname:
    #             newname = "VOD"
    #         else:
    #             pass
    #         # newname = colname[:-3]
    #         df_csv= df_csv.rename(columns={colname: newname})
        
    
    """ #(pixel単位で)月平均とstdで zscoring""" 
    months = [m+1 for m in range(12)]
    vars_list = df_csv.columns.tolist()
    
    df_csv_z = df_csv.copy()
    
    ### obtain monthly mean and std as dict
    monthly_mean_std_dic = {}
    for var in vars_list:
        df_csv_z[f"{var}z"] = np.nan
        df_var = df_csv_z.loc[:,var]
        month_dic = {}
        for m in months:
            specific_month_rows = df_var[df_var.index.month == m]
            monthly_mean = specific_month_rows.mean(skipna=True)
            monthly_std = specific_month_rows.std(skipna=True)
            month_dic[m] = [monthly_mean, monthly_std]
            
            ##インデックスで抽出して元のデータフレームの新規列にzscoreを入れる
            specific_month_idx = specific_month_rows.index.tolist()
            df_csv_z.loc[specific_month_idx, f"{var}z"] = (df_csv_z[var]-monthly_mean)/monthly_std
            # df_csv_z.loc[specific_month_idx, f"{var}z"] = df_csv_z[var]-monthly_mean
    
        monthly_mean_std_dic[var] = month_dic #確認用
        
    
    ### nan削除
    df_valid = df_csv_z.dropna() #nanがあれば除外
    df_valid = df_valid.drop(vars_list, axis=1)
    
    #実験　並び替えてみる
    # df_nono = df_nono[["GSMap","SM","VOD","GEb","GEt","Beneficial","Temperature","Balnce","GPP","WUE"]]
    
    #  normalizing features
    # df_norm = (df_valid - df_valid.mean(axis=0))/df_valid.std(axis=0)
    
    
    """　#Select Dataframe """
    # Y_norm = df_norm['GPPz'] #df_norm #df_valid
    # X_norm = df_norm.drop(['GPPz'], axis=1) #df_norm #df_valid
    Y_norm = df_valid['GOSIFz'] #monthly z scoringしたものを使う
    X_norm = df_valid.drop(['GOSIFz'], axis=1)
    
    varzs = [col for col in X_norm.columns]
    # dfによってvarの順番が変わるためsortする
    # varz_list = sorted(vars_ori)
    
    
    """#PCA"""
    #まずPCAしてから
    # pca = PCA() #次元数指定なし
    # pca.fit(X_norm)
    # X_pca = pca.fit_transform(X_norm) #transformは主成分スコア
    
    # # converting to dataframe
    # names = [f"PC{i+1}" for i in range(X_pca.shape[1])] #PC1~n
    # X_pcadf = pd.DataFrame(X_pca, columns=names)
    
    # #やらなくてもいい
    # # create covariance matrix
    # corr_matrix = np.corrcoef(X_norm.T) #相関係数を求める
    
    # # # create heatmap
    # # plt.figure(figsize=(6, 6))
    # # sns.heatmap(corr_matrix, cmap='magma', annot=True, fmt='.2f',
    # #             xticklabels=X_norm[vars].columns, yticklabels=X_norm[vars].columns)
    # # plt.title('Feature Correlation Heatmap')
    # # plt.show()
    # # If there are some moderate correlations, PCA might be effective in dimensionality reduction.
    
    # # calculate eigenvectors and eigenvalues
    # eigenvalues, eigenvectors = np.linalg.eig(corr_matrix) #合っているかわからなくなった。共分散行列から求めるべき
    # # cov_mat = np.cov(X_norm.T)  # 共分散行列を計算 →normalizedすると結果同じになる
    # # eigenvalues_, eigenvectors_ = np.linalg.eig(cov_mat)
    
    # # sort the eigenvalues and eigenvectors in descending order
    # idx = eigenvalues.argsort()[::-1]
    # eigenvalues_sort = eigenvalues[idx]
    
    # # idx #変数の順番変えるとPC5以降あたりかわる
    
    # """#Contribution"""
    # # convert to dataframes
    # new_var_order = [X_norm[varzs].columns[i] for i in idx]
    # new_order = [names[i] for i in idx]
    # # eigenvalues_df = pd.DataFrame({'Eigenvalue': eigenvalues_sort}, index=anime1[num_cols].columns)
    # eigenvalues_df = pd.DataFrame({'Eigenvalue': eigenvalues_sort}, index=names) #こういう理解
    # eigenvalues_df['Proportion'] = eigenvalues_df['Eigenvalue'] / eigenvalues_df['Eigenvalue'].sum()
    # eigenvalues_df['Cumulative Proportion'] = eigenvalues_df['Proportion'].cumsum()
    
    # print("Eigenvalues:")
    # eigenvalues_df.style.format({'Eigenvalue': '{:.4f}', 'Proportion': '{:.4f}', 'Cumulative Proportion': '{:.4f}'})#分散の比較と同値ぽい
    # インデックスをPC1~PC8に変更
    # 変数の順場を変えるとPC5くらい以降順位かわる
    
    #寄与率でも(各主成分が持つ分散の比率、固有値の割合でもあるぽい) #結果同じ
    # contribution = pca.explained_variance_ratio_
    # contri_df = pd.DataFrame({'Contribution': contribution})
    # contri_df['Proportion'] = contri_df['Contribution'] / contri_df['Contribution'].sum()
    # contri_df['Cumulative Proportion'] = contri_df['Proportion'].cumsum()
    # print("Contribution:")
    # display(contri_df.style.format({'Contribution': '{:.4f}', 'Proportion': '{:.4f}', 'Cumulative Proportion': '{:.4f}'})) #分散の比較と同値ぽい
    
    #ネットよりhttps://www.takapy.work/entry/2019/02/08/002738　各成分の寄与率で、大きい順になるはず
    # print('explained variance ratio: {}'.format(pca.explained_variance_ratio_)) #このコードでsortした固有値と同じになる
    
    # plot scree plot
    # plt.plot(range(1, len(eigenvalues_sort) + 1), eigenvalues_sort, marker='o', color='#8080ff')
    # plt.xlabel('Principal Component')
    # plt.ylabel('Eigenvalue')
    # plt.title('Scree Plot')
    # plt.show()
    
    """#Loadings"""
    num_pc =4 #何成分使うか
    pca = PCA(n_components= num_pc)
    X_pca = pca.fit_transform(X_norm) #fit_transformは主成分スコア算出
    pca.fit(X_norm)
    # X_pca.shape #次元圧縮（n）されたことがわかる
    names = [f"PC{i+1}" for i in range(X_pca.shape[1])] #PC1~n
    
    # pca.explained_variance_ #eigenと同じ
    # pca.components_:分散共分散行列に対する固有ベクトル, 行：成分数、列：元データ次元 主成分の数だけある
    # pca.explained_variance_ #固有値
    # pca.explained_variance_ratio_ : 寄与率(各主成分が持つ分散の比率、固有値の割合でもあるぽい)
    
    load_vals = pca.components_*np.c_[np.sqrt(pca.explained_variance_)] # pca.components_:主成分ベクトル*pca.explained_variance_:分散
    loadings = pd.DataFrame(load_vals.T[:,:num_pc], index=np.array(varzs), columns=names[:num_pc]) #pca.components_.T[:,:3] #Kaggleから変更 #4成分で表す
    
    """# Axis Rotation : Varimax Orthogonal
    けっきょくこれだけで良かった？
    """
    # fit factor analyzer with principal components and varimax rotation
    # fa = FactorAnalyzer(rotation="varimax", n_factors=num_pc, method='principal')
    # fa.fit(X_norm)
    # # fa.loadings_
    
    # # get the rotated factor pattern
    # loadings2 = pd.DataFrame(fa.loadings_, index=X_norm.columns, columns=[f"Factor{i+1}" for i in range(num_pc)])
    # rotated_factor_pattern = loadings2[abs(loadings2) >= 0.5].dropna(how='all')
    
    #これが2次元や3次元だったら変数の分布を成分次元でプロットできる、ってことだと思う
    
    """# Regression"""
    # kaggleのだとRotationしたscore?を使用しているのでやめる→rotateは不要という理解。ぜんぶ足すから。
    scaler = StandardScaler()
    X_pca = pca.fit_transform(X_norm)
    # Scalor使ってる: https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html
    X_pca_scale = scaler.fit_transform(X_pca) #さらにscoreを正規化する
    
    # # scaleする前のscoreで線形回帰
    # model = Ridge() #線形
    # model.fit(X_pca, Y_norm)
    # model.coef_
    #array([ 0.29507563,  0.28495811, -0.07459416, -0.14274526])
    #Ridgeのsummaryの見方がわからない
    
    # #scale後のscoreで線形回帰
    # model = Ridge() #線形
    # model.fit(X_pca_scale, Y_norm)
    # model.coef_
    
    # # scaleする前のscoreで線形回帰
    # X1 = sm.add_constant(X_pca)
    # model_sm = sm.OLS(Y_norm, X1).fit() #線形
    # print(model_sm.summary()) #Ridgeとだいたい同じ
    
    # scale後のscoreで線形回帰
    X1_scale = sm.add_constant(X_pca_scale)
    model_sm_scale = sm.OLS(Y_norm, X1_scale).fit() #線形
    # print(model_sm_scale.summary()) #Ridgeとだいたい同じ
    
    """# Cal relative importance of variables"""
    
    #論文: we multiplied the loading scores of each variable by the PCR coefficients and summed these scores. This enabled us to estimate the relative importance of each variable
    #各変数においてcoefficientsとPC*ごとのloading(相関を意味する)をかけて、それらをPC*分sumする
    coefficients_ori = model_sm_scale.params
    coefficients = coefficients_ori[1:]
    arr_coef = np.array(coefficients)
    
    # p値が有意なcomponentのcoefみ採用
    p_list = model_sm_scale.pvalues[1:].tolist()
    valid_p_idx = [i for i,v in enumerate(p_list) if v < pval] #0.05 #0.1
    
    valid_arr_coef = arr_coef[valid_p_idx]
    
    # p値が有意(0.05)なcomponentのloadingみ採用
    valid_loadings = loadings.iloc[:,valid_p_idx]
    
    #sum (loadings*coefficients)
    importance_dic = {}
    for v in varzs:
      arr_load = np.array(valid_loadings.loc[v,:])
      arr_importance = arr_load*valid_arr_coef #符号はそのままでいいかな（論文はそのまま）script_vsiFun.Rのline179
                                               #script_ORA.Rのline215でabsVal = TRUEにしてるとこで絶対値と思う
      # arr_importance = np.abs(arr_load*valid_arr_coef) #絶対値
      importance = arr_importance.sum()
      importance_dic[v] = importance
    
    importance_dic["r_square"] = model_sm_scale.rsquared_adj
    importance_dic["p_values"] = model_sm_scale.pvalues[1:].tolist()
    
    ### keyの名前変更
    new_vars = [n[:-1] for n in varzs]
    key_mapping = [[o,n] for o,n in zip(varzs,new_vars)]
    key_mapping_dic = dict(key_mapping)
    importance_dic_fin = {key_mapping_dic.get(old_key, old_key): value for old_key, value in importance_dic.items()}
    
    return importance_dic_fin

"""#Chart"""

# vars_rename = []
# for v in vars:
#   if v == "Balnce":
#     new = "Balance"
#     vars_rename.append(new)
#   elif v =="GEb":
#     new = "Evaporation"
#     vars_rename.append(new)
#   elif v == "GEt":
#     new = "Transpiration"
#     vars_rename.append(new)
#   elif v == "GSMap":
#     new = "Precipitation"
#     vars_rename.append(new)
#   elif v == "SM":
#     new = "SoilMoisture"
#     vars_rename.append(new)
#   else:
#     vars_rename.append(v)
# vars_rename


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