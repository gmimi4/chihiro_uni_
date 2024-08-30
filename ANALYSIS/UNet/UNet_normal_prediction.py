# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:08:52 2024
# Google Colab can't run so do on Mac

@author: wtakeuchi
mps: https://zenn.dev/hidetoshi/articles/20220731_pytorch-m1-macbook-gpu
"""


import os
import time
import copy
from collections import defaultdict
import torch
import shutil
import pandas as pd
from skimage import io, transform
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm as tqdm

from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
import cv2

from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.nn.functional as F
from PIL import Image
from PIL import TiffTags
from torch import nn
# import zipfiles

import random
from natsort import natsorted
import glob
import rasterio

device = torch.device('mps')

input_dir = '/Volumes/PortableSSD/Malaysia/01_Blueprint/Pegah_san/03_UNet/1_tiles/imagesTiff'
output_dir = '/Volumes/PortableSSD/Malaysia/01_Blueprint/Pegah_san/03_UNet/OUT_normal'

file_list = glob.glob(input_dir+ "/*.tif")

#作成したモデルを読み込みます(上のdefとmodel = UNet(3,1).cuda()のところを回す)
model_dir = '/Volumes/PortableSSD/Malaysia/01_Blueprint/Pegah_san/03_UNet/2_retraining/1_training_dataset/model'
checkpoint_path = model_dir + os.sep + 'chkpoint_normal.pth' #chkpoint_
best_model_path =  model_dir + os.sep + 'bestmodel_normal.pth' #bestmodel.pt


#U-Netのモデルの定義
class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        # 資料中の『FCN』に当たる部分
        self.conv1 = conv_bn_relu(input_channels,64)
        self.conv2 = conv_bn_relu(64, 128)
        self.conv3 = conv_bn_relu(128, 256)
        self.conv4 = conv_bn_relu(256, 512)
        self.conv5 = conv_bn_relu(512, 1024)
        self.down_pooling = nn.MaxPool2d(2)

        # 資料中の『Up Sampling』に当たる部分
        self.up_pool6 = up_pooling(1024, 512)
        self.conv6 = conv_bn_relu(1024, 512)
        self.up_pool7 = up_pooling(512, 256)
        self.conv7 = conv_bn_relu(512, 256)
        self.up_pool8 = up_pooling(256, 128)
        self.conv8 = conv_bn_relu(256, 128)
        self.up_pool9 = up_pooling(128, 64)
        self.conv9 = conv_bn_relu(128, 64)
        self.conv10 = nn.Conv2d(64, output_channels, 1)

        # nn.init.kaiming_normal_: パラメータの初期化らしい
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # 正規化
        x = x/255.

        # 資料中の『FCN』に当たる部分
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        # 資料中の『Up Sampling』に当たる部分, torch.catによりSkip Connectionをしている
        p6 = self.up_pool6(x5)
        x6 = torch.cat([p6, x4], dim=1)
        x6 = self.conv6(x6)

        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.conv7(x7)

        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.conv8(x8)

        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x9 = self.conv9(x9)

        output = self.conv10(x9)
        output = torch.sigmoid(output)

        return output
    
def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels), #正規化らしい
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
    )

def down_pooling():
    return nn.MaxPool2d(2)

def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        #転置畳み込み
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()




#<---------------各インスタンス作成---------------------->
model = UNet(3,1).to(device) #mps dekiteru?
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)

model, optimizer, start_epoch, valid_loss_min = load_ckp(best_model_path, model, optimizer)


#学習済みモデルは、(Batch_size, 3, H, W) のインプットが想定されています。 (10,3,256,256)と思う
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for input_path in file_list:
  # 前処理
  img_org = Image.open(input_path).convert('RGB')
  img = preprocess(img_org)
  # (3, H, W) -> (1, 3, H, W)
  img_batch = img.unsqueeze(0)
  # print(img.size())
  # print(img_batch.size())
  # 推論

  test_dataloader = DataLoader(img_batch, batch_size=10) #,shuffle=True

  for data in test_dataloader:
        data = torch.autograd.Variable(data, volatile=True).to(device)
        o = model(data)
        tm=o[0][0].data.cpu().numpy() #arrayになる
  #check
   # figure, ax = plt.subplots()
   # ax.imshow(tm, interpolation="nearest", cmap="gray")

  out_path = output_dir + os.sep + os.path.basename(input_path)

  with rasterio.open(input_path) as src:
      profile = src.profile
      transform =  src.transform

      profile_new = {
        'driver': 'GTiff',  # GeoTIFF format
        'height': profile["height"],  # Number of rows
        'width': profile["width"],   # Number of columns
        'count': 1,  # Number of bands (e.g., 1 for grayscale)
        'dtype': profile["dtype"],  # Data type of the array
        'crs': profile["crs"],  # Coordinate Reference System (adjust as needed)
        'transform': transform  # Spatial transformation
    }

      with rasterio.open(out_path, 'w', **profile_new) as dst:
        dst.write(tm,1)




