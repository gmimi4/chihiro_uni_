# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:03:05 2023

@author: chihiro
"""

import os
import glob
import numpy as np
# import tifffile as tiff
import cv2
from skimage import io
import matplotlib.pyplot as plt

# anno_dir = '/Volumes/PortableSSD/Malaysia/01_Blueprint/Pegah_san/03_UNet/2_retraining/1_training_dataset/anno/'
# new_dir = '/Volumes/PortableSSD/Malaysia/01_Blueprint/Pegah_san/03_UNet/2_retraining/1_training_dataset/annoBi'
anno_dir = '/Volumes/PortableSSD/Malaysia/01_Blueprint/SDGuthrie/03_UNet/_retraining/1_training_dataset/anno'
new_dir = '/Volumes/PortableSSD/Malaysia/01_Blueprint/SDGuthrie/03_UNet/_retraining/1_training_dataset/annoBi'

img_files = glob.glob(anno_dir+"/*.tif")

target_file_list = []
for i in img_files:
    # i='/Volumes/PortableSSD/Malaysia/01_Blueprint/Pegah_san/03_UNet/2_retraining/1_training_dataset/36.tif'
    # large_image_stack = tiff.imread(i)
    large_image_stack = io.imread(i)
    img_array = np.array(large_image_stack, dtype='uint8')
    img_array_new = np.where(img_array != 10, 0, img_array) #in Swin, 10 is terrace

    filenmae = os.path.basename(i)
    outfile = new_dir + os.sep + filenmae
    
    io.imsave(outfile, img_array_new)



#check
imgs = glob.glob(new_dir + os.sep + "*.tif")
file_path = imgs[90]

# Read the .tif image
image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

plt.figure(figsize=(10, 10))
plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
plt.colorbar(label="Pixel Intensity" if len(image.shape) == 2 else None)
plt.title("TIF Image")
plt.axis("off")
plt.show()