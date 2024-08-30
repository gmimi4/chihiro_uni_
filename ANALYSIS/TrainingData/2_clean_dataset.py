# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:03:05 2023

@author: chihiro
"""

import os
import glob
import numpy as np
import tifffile as tiff
import cv2

img_dir = r"E:\Malaysia\01_Blueprint\Pegah_san\03_UNet\2_retraining\1_training_dataset\img"
anno_dir = r"E:\Malaysia\01_Blueprint\Pegah_san\03_UNet\2_retraining\1_training_dataset\anno"

img_files = glob.glob(img_dir+"\\*.tif")

target_file_list = []
for i in img_files:
    # i=img_files[9]
    large_image_stack = tiff.imread(i)
    img_array = np.array(large_image_stack, dtype='uint8')
    if np.max(img_array) == 0:
        target_file_list.append(i)
    else:
        pass

print(len(target_file_list))

target_file_list_anno = [a.replace("img", "anno") for a in target_file_list]

for i in target_file_list:
    os.remove(i)
    
for i in target_file_list_anno:
    os.remove(i)



