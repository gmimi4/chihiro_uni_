# -*- coding: utf-8 -*-
# """
# Created on Wed Aug 17 17:06:48 2022

# @author: Haidar
# work dir: C:\Users\Haidar\Documents\AOI1
# """

#IMAGE
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
import cv2
import os

# whole_img_path = r"D:\Malaysia\01_Brueprint\09_Classification\0_preparation\ras\img_clip.tif"
# whole_anno_path = r"D:\Malaysia\01_Brueprint\09_Classification\0_preparation\ras\label_clip.tif"
# out_dir_parent = r"D:\Malaysia\01_Brueprint\09_Classification\1_training_dataset"

whole_img_path = r"E:\Malaysia\01_Blueprint\Pegah_san\03_UNet\2_retraining\0_preparation\CS_clip_extent.tif"
# whole_img_path = r"D:\Malaysia\01_Brueprint\11_Roads\04_UNet_road\01_preparation\Curvature_Diss_Slope05_Diss_Homo_8bi_extentTrain.tif"
# whole_img_path = r"D:\Malaysia\01_Brueprint\11_Roads\04_UNet_road\01_preparation\Curvature_Diss_SlopeR_Diss_Homo_8bi_extentTrain.tif"
whole_anno_path = r"E:\Malaysia\01_Blueprint\Pegah_san\03_UNet\2_retraining\0_preparation\2_convert\polygon_ras.tif"
out_dir_parent = r"E:\Malaysia\01_Blueprint\Pegah_san\03_UNet\2_retraining\1_training_dataset"

out_img_dir = os.path.join(out_dir_parent,"img")
out_anno_dir = os.path.join(out_dir_parent,"anno")
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_anno_dir, exist_ok=True)


cells = 64
steps = 32

large_image_stack = tiff.imread(whole_img_path)
img_array = np.array(large_image_stack, dtype='uint8')

patches_img = patchify(img_array,(cells,cells,3),step=steps) #all 256
# patches_img = patchify(img_array,(512,512,3),step=512) #32 is the number of desiered overlapping pixels
print(patches_img.shape)
patches_img = np.squeeze(patches_img)
print(patches_img.shape)
img = 0
    
for i in range(patches_img.shape[0]):
 for j in range(patches_img.shape[1]):
            
  patch = patches_img[i,j,:,:,:]
  # tiff.imwrite(r"D:\Malaysia\01_Brueprint\09_Classification\1_training_dataset\img/" + str(img) + ".tif", patch) #\tiles/
  # tiff.imwrite(out_dir_parent + "\\img\\" + str(img) + ".tif", patch)
  tiff.imwrite(out_img_dir +os.sep + str(img) + ".tif", patch)
  img +=1 
  

#LABEL  
import cv2
import numpy as np
from patchify import patchify
from matplotlib import pyplot as plt
import tifffile as tiff

large_mask_stack = tiff.imread(whole_anno_path)
label_array = np.array(large_mask_stack, dtype='uint8')

patches_label = patchify(label_array,(cells,cells),step=steps) #32 is the number of desiered overlapping pixels
print(patches_label.shape)
#patches_img = np.squeeze(patches_img)
#print(patches_img.shape)
img = 0

for i in range(patches_label.shape[0]):
 for j in range(patches_label.shape[1]):
            
  patch = patches_label[i,j,:]
  # tiff.imwrite("D:\Malaysia\01_Brueprint\09_Classification\1_training_dataset\anno/" + str(img) + ".tif", patch)
  # tiff.imwrite(out_dir_parent + "\\anno\\" + str(img) + ".tif", patch)
  tiff.imwrite(out_anno_dir + os.sep + str(img) + ".tif", patch)
  #plt.imsave('AOI_tes2/label_binary_patches/' + str(img) + ".jpg", patch, cmap='gray') #hanya utk save jpg supaya terlihat jelas
  img +=1 


# #Cek hasil secara visual
# import rasterio as rio  
# from rasterio.plot import show

# img = rio.open('label_patches/478.tif')
# show(img) # plot raster
# img_array = img.read()
# print(img_array.shape)
