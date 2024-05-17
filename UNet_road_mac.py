#!/usr/bin/env python3
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

parent_dir = '/Volumes/PortableSSD/Malaysia/01_Blueprint/04_UNet_road/02_training/SLOPE0R_composit' #SLOPE05m_composit #SLOPE05m_Homo #SLOPE0R_composit
img_dir = os.path.join(parent_dir,'img')
ano_dir = os.path.join(parent_dir,'anno')

img_listt = natsorted(glob.glob(os.path.join(img_dir,"*.tif")))
ano_listt = natsorted(glob.glob(os.path.join(ano_dir,"*.tif")))
filenames = [s[:-4] for s in os.listdir(img_dir)]

""" #Augumentation """
#画像データ拡張の関数
# albumentations.normalization: https://albumentations.ai/docs/api_reference/augmentations/transforms/
def get_train_transform():
   return A.Compose(
       [
        #リサイズ(こちらはすでに適用済みなのでなくても良いです)
        A.Resize(256, 256),
        #正規化(こちらの細かい値はalbumentations.augmentations.transforms.Normalizeのデフォルトの値を適用)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #水平フリップ（pはフリップする確率）
        A.HorizontalFlip(p=0.25),
        #垂直フリップ
        A.VerticalFlip(p=0.25),
        ToTensorV2()
        ])


#Datasetクラスの定義
class LoadDataSet(Dataset):
        def __init__(self,path, transform=None):
            self.path = path
            self.folders = os.listdir(path) #['anno', 'img']
            self.transforms = get_train_transform() #Augumentationで定義した
            self.image_folder = os.path.join(self.path, self.folders[1])

        # __len__: 組み込み関数？ここでは独自に定義？　https://blog.codecamp.jp/python-class-code
        def __len__(self):
            self.image_folder = os.path.join(self.path, self.folders[1])
            self.image_list =os.listdir(self.image_folder)
            # return len(self.image_folder)
            return len(self.image_list)


        def __getitem__(self,idx): #idx: indexじゃないとだめぽい
            self.folders = os.listdir(self.path) #['img', 'anno']
            image_folder = os.path.join(self.path, self.folders[0]) #[1]
            mask_folder = os.path.join(self.path, self.folders[1]) #[0]
            image_path = os.path.join(image_folder,natsorted(os.listdir(image_folder))[idx])
            # image_path = '/Volumes/PortableSSD/Malaysia/01_Blueprint/04_UNet_road/02_training/SLOPE05m_composit/img/8.tif'

            #画像データの取得
            img = io.imread(image_path)[:,:,:3].astype('float32')
            img = transform.resize(img,(256,256))

            mask = self.get_mask(mask_folder, 256, 256, idx).astype('float32')

            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            # mask = mask[0].permute(2, 0, 1)
            mask = mask.permute(2, 0, 1)
            return (img,mask)


        #マスクデータの取得
        def get_mask(self,mask_folder,IMG_HEIGHT, IMG_WIDTH, idx):
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1)) #dtype=np.bool
            # for mask_ in os.listdir(mask_folder): #ファイル名リスト
            #         mask_ = io.imread(os.path.join(mask_folder,mask_))
            #         mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH)) #(256, 256)にする
            #         mask_ = np.expand_dims(mask_,axis=-1) #np.expand_dims指定位置に次元追加（）shapeが変わる
            #         mask = np.maximum(mask, mask_)

            # return mask
            # mask_ = [s for s in os.listdir(mask_folder) if str(idx) in s][0] #ファイル名リストからidxを含むファイル名のを取り出す
            mask_ = os.path.join(mask_folder,natsorted(os.listdir(mask_folder))[idx])
            mask_ = io.imread(os.path.join(mask_folder,mask_))
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH)) #(256, 256)にする
            mask_ = np.expand_dims(mask_,axis=-1) #np.expand_dims指定位置に次元追加（）shapeが変わる
            mask = np.maximum(mask, mask_)

            return mask

TRAIN_PATH = parent_dir
path = TRAIN_PATH
train_dataset = LoadDataSet(TRAIN_PATH, transform=get_train_transform())

model_dir = os.path.join(parent_dir, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

#画像枚数を確認します（回す必要あり）
num_of_sample = train_dataset.__len__()

#入力画像とマスクのデータがどうなっているのか確認してみます。
def format_image(img):
    img = np.array(np.transpose(img, (1,2,0)))
    #下は画像拡張での正規化を元に戻しています
    mean=np.array((0.485, 0.456, 0.406))
    std=np.array((0.229, 0.224, 0.225))
    img  = std * img + mean
    img = img*255
    img = img.astype(np.uint8)
    return img

def format_mask(mask):
    # mask = np.squeeze(np.transpose(mask, (1,2,0))) #np.squeeze: 引数がなければ配列が1しかない次元を削除
    mask = np.transpose(mask, (1,2,0))
    return mask

def visualize_dataset(n_images, predict=None):
  images = random.sample(range(0, num_of_sample), n_images) #n_imagesは表示したい数, num_of_sampleはファイル数
  # images = random.sample(filenames, n_images) #n_imagesは表示したい数, num_of_sampleはファイル数
  figure, ax = plt.subplots(nrows=len(images), ncols=2, figsize=(5, 8)) #imgesは選定したid
  print(images) #ファイル名（.tifなし）リスト
  for i in range(0, len(images)):
    img_no = images[i] #idを得る
    image, mask = train_dataset.__getitem__(int(img_no))
    image = format_image(image)
    mask = format_mask(mask)
    ax[i, 0].imshow(image)
    ax[i, 1].imshow(mask, interpolation="nearest", cmap="gray")
    ax[i, 0].set_title("Input Image")
    ax[i, 1].set_title("Label Mask")
    ax[i, 0].set_axis_off()
    ax[i, 1].set_axis_off()
  plt.tight_layout()
  plt.show()
  figure.savefig(model_dir + os.sep + "training_imgs.png")

visualize_dataset(5)



""" #Training """
#データの前処理。
split_ratio = 0.3
train_size=int(np.round(train_dataset.__len__()*(1 - split_ratio),0))
valid_size=int(np.round(train_dataset.__len__()*split_ratio,0))
train_data, valid_data = random_split(train_dataset, [train_size, valid_size])
train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=True) #batch size10 ni shiteru
val_loader = DataLoader(dataset=valid_data, batch_size=10)

print("Length of train　data: {}".format(len(train_data)))
print("Length of validation　data: {}".format(len(valid_data)))

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

#畳み込みとバッチ正規化と活性化関数Reluをまとめている
# nn.Sequentialは「モデルとして組み込む関数を1セットとして順番に実行しますよ」というもの
# nn.BatchNorm2d: https://jvgd.medium.com/pytorch-batchnorm2d-weights-explained-13705ac21189
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

#損失関数について(力尽きてまだ十分確認できていない)
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice


#IoUのクラスを定義
class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return IoU
    
#別のコラボだった
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


#U-Netの学習を行います(GPU onにする)
#<---------------各インスタンス作成---------------------->
# model = UNet(3,1).cuda() #GPUでやる
model = UNet(3,1).to(device) #mps dekiteru?

optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
criterion = DiceLoss()
accuracy_metric = IoU()
num_epochs=100 #84でコラボの使用上限に達した
valid_loss_min = np.Inf

checkpoint_path = os.path.join(model_dir, 'chkpoint.pth') #chkpoint_
best_model_path = os.path.join(model_dir, 'bestmodel.pth') #bestmodel.pt

total_train_loss = []
total_train_score = []
total_valid_loss = []
total_valid_score = []


losses_value = 0
for epoch in range(num_epochs):
  #<---------------トレーニング---------------------->
    train_loss = []
    train_score = []
    valid_loss = []
    valid_score = []
    pbar = tqdm(train_loader, desc = 'description')
    for x_train, y_train in pbar:
      # x_train = torch.autograd.Variable(x_train).cuda()
      # y_train = torch.autograd.Variable(y_train).cuda()
      x_train = torch.autograd.Variable(x_train).to(device)
      y_train = torch.autograd.Variable(y_train).to(device)
      optimizer.zero_grad()
      output = model(x_train)
      ## 損失計算
      loss = criterion(output, y_train)
      losses_value = loss.item()
      ## 精度評価
      score = accuracy_metric(output,y_train)
      loss.backward()
      optimizer.step()
      train_loss.append(losses_value)
      train_score.append(score.item())
      pbar.set_description(f"Epoch: {epoch+1}, loss: {losses_value}, IoU: {score}")
    #<---------------評価---------------------->
    with torch.no_grad():
      for image,mask in val_loader:
        # image = torch.autograd.Variable(image).cuda()
        # mask = torch.autograd.Variable(mask).cuda()
        image = torch.autograd.Variable(image).to(device)
        mask = torch.autograd.Variable(mask).to(device)
        output = model(image)
        ## 損失計算
        loss = criterion(output, mask)
        losses_value = loss.item()
        ## 精度評価
        score = accuracy_metric(output,mask)
        valid_loss.append(losses_value)
        valid_score.append(score.item())

    total_train_loss.append(np.mean(train_loss))
    total_train_score.append(np.mean(train_score))
    total_valid_loss.append(np.mean(valid_loss))
    total_valid_score.append(np.mean(valid_score))
    print(f"Train Loss: {total_train_loss[-1]}, Train IOU: {total_train_score[-1]}")
    print(f"Valid Loss: {total_valid_loss[-1]}, Valid IOU: {total_valid_score[-1]}")

    checkpoint = {
        'epoch': epoch + 1,
        'valid_loss_min': total_valid_loss[-1],
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    # checkpointの保存
    save_ckp(checkpoint, False, checkpoint_path, best_model_path)

    # 評価データにおいて最高精度のモデルのcheckpointの保存
    if total_valid_loss[-1] <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,total_valid_loss[-1]))
        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = total_valid_loss[-1]

    print("")
    
    
#U-Netモデルの性能評価の確認 #by matplotlib
# import seaborn as sns

plt.figure(1)
plt.figure(figsize=(15,5))
# sns.set_style(style="darkgrid")
plt.subplot(1, 2, 1)
# sns.lineplot(x=range(1,num_epochs+1), y=total_train_loss, label="Train Loss")
# sns.lineplot(x=range(1,num_epochs+1), y=total_valid_loss, label="Valid Loss")
plt.plot(range(1,num_epochs+1), total_train_loss, label="Train Loss")
plt.plot(range(1,num_epochs+1), total_valid_loss, label="Valid Loss")
plt.title("Loss")
plt.xlabel("epochs")
plt.ylabel("DiceLoss")

plt.subplot(1, 2, 2)
plt.plot(range(1,num_epochs+1), total_train_score, label="Train Score")
plt.plot(range(1,num_epochs+1), total_valid_score, label="Valid Score")
plt.title("Score (IoU)")
plt.xlabel("epochs")
plt.ylabel("IoU")
plt.show()


#作成したモデルを読み込みます
model, optimizer, start_epoch, valid_loss_min = load_ckp(best_model_path, model, optimizer)

#続いて入力画像と教師データ、モデルによる出力を表示する関数を用意し、出力を行います。
def visualize_predict(model, n_images):
  figure, ax = plt.subplots(nrows=n_images, ncols=3, figsize=(10, 12))
  with torch.no_grad():
    for data,mask in val_loader:
        data = torch.autograd.Variable(data, volatile=True).to(device)
        mask = torch.autograd.Variable(mask, volatile=True).to(device)
        o = model(data)
        break
  for img_no in range(0, n_images):
    tm=o[img_no][0].data.cpu().numpy()
    img = data[img_no].data.cpu()
    msk = mask[img_no].data.cpu()
    img = format_image(img)
    msk = format_mask(msk)
    ax[img_no, 0].imshow(img)
    ax[img_no, 1].imshow(msk, interpolation="nearest", cmap="gray")
    ax[img_no, 2].imshow(tm, interpolation="nearest", cmap="gray")
    ax[img_no, 0].set_title("Input Image")
    ax[img_no, 1].set_title("Label Mask")
    ax[img_no, 2].set_title("Predicted Mask")
    ax[img_no, 0].set_axis_off()
    ax[img_no, 1].set_axis_off()
    ax[img_no, 2].set_axis_off()
  plt.tight_layout()
  plt.show()

visualize_predict(model, 6)


