# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 17:06:48 2022

@author: Haidar
work dir: C:\Users\Haidar\Documents\AOI1
rasterio_copy env
"""

#IMAGE
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
import cv2
from PIL import Image

whole_img_path = r"F:\MAlaysia\Blueprint\POC_Area_20230822\Image_Asahan_Terrace\AsahanTer1_resample20cm.tif"
out_dir_parent = r"D:\Malaysia\01_Brueprint\11_Roads\01_GLCM\01_preparation\img_clip"

cells = 512
steps = 512

large_image_stack = tiff.imread(whole_img_path)
img_array = np.array(large_image_stack, dtype='uint8')

patches_img = patchify(img_array,(cells,cells,3),step=steps) #all 256
# patches_img = patchify(img_array,(512,512,3),step=512) #32 is the number of desiered overlapping pixels
print(patches_img.shape)
patches_img = np.squeeze(patches_img)
print(patches_img.shape)

#以下、たぶんどっちでもいい
#　https://levelup.gitconnected.com/how-to-split-an-image-into-patches-with-python-e1cf42cf4f77
# for i in range(patches_img.shape[0]):
#     for j in range(patches_img.shape[1]):
#         patch = patches_img[i, j, 0]
#         patch = Image.fromarray(patch)
#         num = i * patches_img.shape[1] + j
#         patch.save(out_dir_parent + "\\img" + str(img) + ".tif")

#元
img = 0
for i in range(patches_img.shape[0]):
  for j in range(patches_img.shape[1]):
       
      patch = patches_img[i,j,:,:,:]
      # tiff.imwrite(r"D:\Malaysia\01_Brueprint\09_Classification\1_training_dataset\img/" + str(img) + ".tif", patch) #\tiles/
      tiff.imwrite(out_dir_parent + "\\img" + str(img) + ".tif", patch)
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
  tiff.imwrite(out_dir_parent + "\\anno\\" + str(img) + ".tif", patch)
  #plt.imsave('AOI_tes2/label_binary_patches/' + str(img) + ".jpg", patch, cmap='gray') #hanya utk save jpg supaya terlihat jelas
  img +=1 


#Cek hasil secara visual
import rasterio as rio  
from rasterio.plot import show

img = rio.open('label_patches/478.tif')
show(img) # plot raster
img_array = img.read()
print(img_array.shape)
