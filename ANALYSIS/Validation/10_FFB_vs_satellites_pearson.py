# -*- coding: utf-8 -*-
"""
Compare overall correlation in YieldWater with satellite monthly pearson
"""
import os
import glob
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import cv2

pp = "_pearson_0"
FFB_ovreall =rf"D:\Malaysia\02_Timeseries\YieldWater\09_Strategy\01_overall_correlation\{pp}\overall_correlation.csv" 
out_dir = r"D:\Malaysia\Validation\7_important_vars\_comparison"

satellite_dir = r"D:\Malaysia\Validation\7_important_vars"
satellite_overalls = glob.glob(satellite_dir + os.sep + "*.csv")


## ------------------------------------------
""" # Liner comparison """ 
## ------------------------------------------
""" create dataset """           
df_ffb = pd.read_csv(FFB_ovreall, index_col=0)

for sate in satellite_overalls:
    df_sate = pd.read_csv(sate, index_col=0)
    satename = os.path.basename(sate)[:-4]
    
    dataset_list = []
    for rege, srow in df_sate.iterrows():
        rege_ = rege.replace(" ","")
        df_ffb_rege = df_ffb.loc[rege_,:]
        srow.name = "satellite"
        df_ffb_rege.name = "FFB"
        df_merge = pd.concat([df_ffb_rege, srow],axis=1)
        dataset_list.append(df_merge)
    
    df_dataset = pd.concat(dataset_list)

    ### Calculate Pearson correlation between two columns
    corr_slp, p_value = pearsonr(df_dataset['satellite'], df_dataset["FFB"])
    
    ## Plot
    plt.figure(figsize=(5, 5))
    plt.scatter(df_dataset["satellite"], df_dataset['FFB'], label=f'FFB vs {satename}')
    # Calculate the least squares fit (linear regression)
    slope, intercept = np.polyfit(df_dataset["satellite"], df_dataset['FFB'], 1)
    regression_line = slope * df_dataset["satellite"] + intercept
    plt.plot(df_dataset["satellite"], regression_line, color='grey',linestyle='--', label='Least Squares Fit')
    plt.xlabel("satellite", fontsize=16)
    plt.ylabel('FFB', fontsize=16)
    # plt.legend()
    plt.text(0.1,0.8,'$ r $=' + str(round(corr_slp, 4)),fontsize=14, transform=plt.gca().transAxes)
    plt.text(0.1,0.7,'$ p $=' + str(round(p_value, 4)),fontsize=14, transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()
    out_fig_dir = out_dir + os.sep + pp
    os.makedirs(out_fig_dir, exist_ok=True)
    plt.savefig(out_fig_dir + os.sep + f"FFB_{satename}.png")
    plt.close()


## うなくぽい結果にならないからやらない
# ## ------------------------------------------
# """ # dHash """
# #https://qiita.com/tishihara/items/977d01445f8b7c6ab3d2
# ## ------------------------------------------

# out_hash_dir = out_dir + os.sep + pp + os.sep + "dhash"
# os.makedirs(out_hash_dir, exist_ok=True)

# """ create correlation image"""
# arr_ffb = df_ffb.to_numpy()

# def correlation_image(arr,outname):
#     colors = ['blue', 'white', 'red']  # Blue for negative, white for zero, red for positive
#     cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
#     norm = Normalize(vmin=arr.min(), vmax=arr.max())
#     plt.imshow(arr, cmap=cmap, norm=norm)
#     output_file = out_hash_dir + os.sep + f"{outname}.png"
#     plt.savefig(output_file, dpi=300)
#     plt.close()
    
# correlation_image(arr_ffb,f"FFB{pp}")

# for sate in satellite_overalls:
#     df_sate = pd.read_csv(sate, index_col=0)
#     satename = os.path.basename(sate)[:-4]
#     arr_sate = df_sate.to_numpy()
#     correlation_image(arr_sate,f"{satename}")
    
    
    
# def dhash(image, hash_size=8):
#     # Convert image to grayscale
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Resize the image to (hash_size + 1, hash_size + 1)
#     resized_image_gray = cv2.resize(image_gray, (hash_size + 1, hash_size + 1), cv2.INTER_AREA)
    
#     # Compute the horizontal and vertical differences
#     dhash_h = resized_image_gray[:-1, 1:] > resized_image_gray[:-1, :-1]
#     dhash_v = resized_image_gray.T[:-1, 1:] > resized_image_gray.T[:-1, :-1]
    
#     # Concatenate both the horizontal and vertical differences
#     dhash = np.concatenate((dhash_h.reshape(-1), dhash_v.reshape(-1)))
#     return dhash

# def hamming_distance(hash1, hash2):
#     # Compute the Hamming distance (number of different bits)
#     return np.count_nonzero(hash1 != hash2)


# image_ffb = cv2.imread(out_hash_dir + os.sep +f"FFB{pp}.png")
# hash1 = dhash(image_ffb)

# ## satelite pngs
# image_satellites = glob.glob(out_hash_dir + os.sep + "*.png")

# hash_result = {}
# for sate in image_satellites:
#     satename = os.path.basename(sate)[:-4]
#     image_sate = cv2.imread(sate)
#     hash2 = dhash(image_sate)
#     # Calculate the Hamming distance between the two hashes
#     distance = hamming_distance(hash1, hash2)
#     hash_result[satename] = distance