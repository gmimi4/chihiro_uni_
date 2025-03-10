#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:54:35 2024

@author: wtakeuchi

# https://github.com/ashish-s-bisht/SwinUnetArchitecturePytorch/blob/main/SwinUnetArchitecturePytorch.ipynb
"""

import os
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import torch
import torch.nn as nn
import torch.nn.functional
import torchvision
from torch.utils.data import TensorDataset
import math
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
from natsort import natsorted
from skimage import io, transform
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import shutil
from torchvision import transforms
from PIL import Image
import rasterio
import random

device = torch.device('mps')
DEVICE = device

dataset_path = '/Volumes/PortableSSD 1/Malaysia/01_Blueprint/SDGuthrie/03_UNet/_retraining/1_training_dataset'
# dataset_path = '/Volumes/PortableSSD/Malaysia/01_Blueprint/Pegah_san/03_UNet/2_retraining/1_training_dataset'
# dataset_path = '/Volumes/PortableSSD 1/MAlaysia/01_Blueprint/Pegah_san/03_UNet/2_retraining/1_training_dataset'
# path = '/Volumes/PortableSSD/Malaysia/01_Blueprint/Pegah_san/03_UNet/2_retraining/1_training_dataset/img/9.tif'
model_dir = '/Volumes/PortableSSD 1/Malaysia/01_Blueprint/SDGuthrie/03_UNet/_retraining/model'
# model_dir = dataset_path + os.sep + 'model'
# model_dir = '/Volumes/PortableSSD 1/MAlaysia/01_Blueprint/Pegah_san/03_UNet/2_retraining/1_training_dataset/model'

img_dir = os.path.join(dataset_path,'img')
ano_dir = os.path.join(dataset_path,'annoBi') #Binary
img_listt = natsorted(glob(os.path.join(img_dir,"*.tif")))
ano_listt = natsorted(glob(os.path.join(ano_dir,"*.tif")))

### Dataset loader
# def load_image(path, size):
#     image = cv2.imread(path)
#     image = cv2.resize(image, (size,size))
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     image = image/255.
#     return image

def get_train_transform(size):
    return A.Compose(
        [
         A.Resize(size, size),
         #正規化(こちらの細かい値はalbumentations.augmentations.transforms.Normalizeのデフォルトの値を適用)
         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
         #水平フリップ（pはフリップする確率）
         A.HorizontalFlip(p=0.5),
         #垂直フリップ
         A.VerticalFlip(p=0.5),
         ToTensorV2()
         ])
    

# def load_data(root_path, size):
#     images = []
#     masks = []
#     img_dir = dataset_path + os.sep + 'img'
#     mask_dir = dataset_path + os.sep + 'anno'
    
#     imgs_path = img_dir + os.sep + '*'
#     masks_path = mask_dir + os.sep + '*'
    
#     for path in sorted(glob(imgs_path)):
#         img = load_image(path, size)
#         images.append(img)
    
#     for path in sorted(glob(masks_path)):
#         img = load_image(path, size)
#         masks.append(img)
        
        
#     return np.array(images), np.array(masks)


class LoadDataSet(Dataset):
        def __init__(self,path, transform=None):
            self.path = path
            self.folders = os.listdir(path) #['anno', 'img']
            self.transforms = get_train_transform(size) #Augumentationで定義した
            # self.image_folder = os.path.join(self.path, self.folders[1])
            self.image_folder = os.path.join(self.path, 'img')

        # __len__: 組み込み関数？ここでは独自に定義？　https://blog.codecamp.jp/python-class-code
        def __len__(self):
            # self.image_folder = os.path.join(self.path, self.folders[1])
            self.image_folder = os.path.join(self.path, 'img')
            self.image_list =os.listdir(self.image_folder)
            # return len(self.image_folder)
            return len(self.image_list)


        def __getitem__(self,idx): #idx: indexじゃないとだめぽい
            self.folders = os.listdir(self.path) #['img', 'anno']
            # image_folder = os.path.join(self.path, self.folders[0]) #[1]
            image_folder = os.path.join(self.path, 'img')
            mask_folder = os.path.join(self.path, 'annoBi') #[0]
            image_path = os.path.join(image_folder,natsorted(os.listdir(image_folder))[idx])
            # image_path = '/Volumes/PortableSSD/Malaysia/01_Blueprint/04_UNet_road/02_training/SLOPE05m_composit/img/8.tif'

            #画像データの取得
            img = io.imread(image_path)[:,:,:3].astype('float32')
            img = transform.resize(img,(size,size))

            mask = self.get_mask(mask_folder, size, size, idx).astype('float32')

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


size = 224 #for resize

train_dataset = LoadDataSet(dataset_path, transform=get_train_transform(size)) #better augmentation only train?

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
  figure, ax = plt.subplots(nrows=len(images), ncols=2, figsize=(5, 8)) #imgesは選定したimg id
  print(images) #ファイル名（.tifなし）リスト
  for i in range(0, len(images)):
    img_no = images[i] #選定したimg idを得る
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
  # figure.savefig(model_dir + os.sep + "training_imgs.png")

visualize_dataset(5)

BATCH_SIZE = 12#12

split_ratio = 0.3
train_size=int(np.round(train_dataset.__len__()*(1 - split_ratio),0))
valid_size=int(np.round(train_dataset.__len__()*split_ratio,0))
train_data, valid_data = random_split(train_dataset, [train_size, valid_size])

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=True)


# X, Y = load_data(dataset_path + os.sep + '*', size)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, shuffle=True)
# X_valid, X_test, Y_valid, Y_test = train_test_split(X_test, Y_test, test_size=0.5,shuffle=True)
# X_train.shape, X_valid.shape, X_test.shape, Y_train.shape, Y_valid.shape, Y_test.shape

# '''
# Note always use pytorch transforms alongside dataloaders for image augmentation.
# This is a crude way of doing thing.
# '''
# X_train = np.expand_dims(X_train,-1)
# Y_train = np.expand_dims(Y_train,-1)
    
# X_train = torch.from_numpy(np.float32(np.squeeze(X_train,-1))).unsqueeze(1)
# Y_train = torch.from_numpy(np.float32(np.squeeze(Y_train,-1))).unsqueeze(1)
# X_valid = torch.from_numpy(np.float32(X_valid)).unsqueeze(1)
# Y_valid = torch.from_numpy(np.float32(Y_valid)).unsqueeze(1)
# X_test = torch.from_numpy(np.float32(X_test)).unsqueeze(1)
# Y_test = torch.from_numpy(np.float32(Y_test)).unsqueeze(1)
# X_train.shape, X_valid.shape, X_test.shape, Y_train.shape, Y_valid.shape, Y_test.shape
    
# train_loader = torch.utils.data.DataLoader(TensorDataset(X_train,Y_train), batch_size=BATCH_SIZE, shuffle=True)

# valid_loader = torch.utils.data.DataLoader(TensorDataset(X_valid,Y_valid), batch_size=BATCH_SIZE, shuffle=True)

# test_loader = torch.utils.data.DataLoader(TensorDataset(X_test,Y_test), batch_size=BATCH_SIZE, shuffle=True)


### Swin-Unet Architecture
def window_partition(x, window_size): #window_size = 7 in SwinTransformerBlock Class
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows

def window_reverse(windows, window_size, H, W): #window_size = (7, 7)
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x

def get_relative_position_index(win_h: int, win_w: int):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid(torch.arange(win_h), torch.arange(win_w),indexing='ij'))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww


class WindowAttention(nn.Module):
    def __init__(
            self,
            dim,
            window_size,
    ):
        super().__init__()
        self.window_size = window_size
        self.window_area = self.window_size[0]*self.window_size[1]
        self.num_heads = 4
        head_dim =  dim // self.num_heads
        # attn_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) **2, self.num_heads))

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(self.window_size[0], self.window_size[1]), persistent=False)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos_bias(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, mask = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)


        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + self._get_rel_pos_bias()
        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
            self,  dim, input_resolution, window_size = 7, shift_size = 0):

        super().__init__()
        self.input_resolution = input_resolution
        window_size = (window_size, window_size)
        shift_size = (shift_size, shift_size)
        self.window_size = window_size
        self.shift_size = shift_size
        self.window_area = self.window_size[0] * self.window_size[1]

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
        )

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.LayerNorm(4 * dim),
            nn.Linear( 4 * dim, dim)
        )

        if self.shift_size:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            H = math.ceil(H / self.window_size[0]) * self.window_size[0]
            W = math.ceil(W / self.window_size[1]) * self.window_size[1]
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None)):
                for w in (
                        slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def _attn(self, x):
        B, H, W, C = x.shape

        # cyclic shift
        if self.shift_size:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        shifted_x = shifted_x[:, :H, :W, :].contiguous()

        # reverse cyclic shift
        if self.shift_size:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        else:
            x = shifted_x
        return x

    def forward(self, x):
        B, H, W, C = x.shape
        B, H, W, C = x.shape
        x = x + self._attn(self.norm1(x))
        x = x.reshape(B, -1, C)
        x = x + self.mlp(self.norm2(x))
        x = x.reshape(B, H,W, C)
        return x
    
    
class PatchEmbedding(nn.Module):
    def __init__(self, in_ch, num_feat, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(in_ch,num_feat, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).permute(0,2,3,1)

class PatchMerging(nn.Module):

    def __init__(
            self,
            dim
    ):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchExpansion(nn.Module):

    def __init__(
            self,
            dim
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim//2)
        self.expand = nn.Linear(dim, 2*dim, bias=False)

    def forward(self, x):

        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H , W, 2, 2, C//4)
        x = x.permute(0,1,3,2,4,5)

        x = x.reshape(B,H*2, W*2 , C//4)

        x = self.norm(x)
        return x

class FinalPatchExpansion(nn.Module):

    def __init__(
            self,
            dim
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, 16*dim, bias=False)

    def forward(self, x):

        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H , W, 4, 4, C//16)
        x = x.permute(0,1,3,2,4,5)

        x = x.reshape(B,H*4, W*4 , C//16)

        x = self.norm(x)
        return x


class SwinBlock(nn.Module):
    def __init__(self, dims, ip_res, ss_size = 3):
        super().__init__()
        self.swtb1 = SwinTransformerBlock(dim=dims, input_resolution=ip_res)
        self.swtb2 = SwinTransformerBlock(dim=dims, input_resolution=ip_res, shift_size=ss_size)

    def forward(self, x):
        return self.swtb2(self.swtb1(x))
    

class Encoder(nn.Module):
    def __init__(self, C, partioned_ip_res, num_blocks=3):
        super().__init__()
        H,W = partioned_ip_res[0], partioned_ip_res[1]
        self.enc_swin_blocks = nn.ModuleList([
            SwinBlock(C, (H, W)),
            SwinBlock(2*C, (H//2, W//2)),
            SwinBlock(4*C, (H//4, W//4))
        ])
        self.enc_patch_merge_blocks = nn.ModuleList([
            PatchMerging(C),
            PatchMerging(2*C),
            PatchMerging(4*C)
        ])

    def forward(self, x):
        skip_conn_ftrs = []
        for swin_block,patch_merger in zip(self.enc_swin_blocks, self.enc_patch_merge_blocks):
            x = swin_block(x)
            skip_conn_ftrs.append(x)
            x = patch_merger(x)
        return x, skip_conn_ftrs
    

class Decoder(nn.Module):
    def __init__(self, C, partioned_ip_res, num_blocks=3):
        super().__init__()
        H,W = partioned_ip_res[0], partioned_ip_res[1]
        self.dec_swin_blocks = nn.ModuleList([
            SwinBlock(4*C, (H//4, W//4)),
            SwinBlock(2*C, (H//2, W//2)),
            SwinBlock(C, (H, W))
        ])
        self.dec_patch_expand_blocks = nn.ModuleList([
            PatchExpansion(8*C),
            PatchExpansion(4*C),
            PatchExpansion(2*C)
        ])
        self.skip_conn_concat = nn.ModuleList([
            nn.Linear(8*C, 4*C),
            nn.Linear(4*C, 2*C),
            nn.Linear(2*C, 1*C)
        ])

    def forward(self, x, encoder_features):
        for patch_expand,swin_block, enc_ftr, linear_concatter in zip(self.dec_patch_expand_blocks, self.dec_swin_blocks, encoder_features,self.skip_conn_concat):
            x = patch_expand(x)
            x = torch.cat([x, enc_ftr], dim=-1)
            x = linear_concatter(x)
            x = swin_block(x)
        return x
    
    
class SwinUNet(nn.Module):
    def __init__(self, H, W, ch, C, num_class, num_blocks=3, patch_size = 4): #num_class=1000 in ori code
        super().__init__()
        self.patch_embed = PatchEmbedding(ch, C, patch_size) #(self, in_ch, num_feat, patch_size)
        self.encoder = Encoder(C, (H//patch_size, W//patch_size),num_blocks)
        self.bottleneck = SwinBlock(C*(2**num_blocks), (H//(patch_size* (2**num_blocks)), W//(patch_size* (2**num_blocks))))
        self.decoder = Decoder(C, (H//patch_size, W//patch_size),num_blocks)
        self.final_expansion = FinalPatchExpansion(C)
        self.head        = nn.Conv2d(C, num_class, 1,padding='same') #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,

    def forward(self, x):
        x = self.patch_embed(x)

        x,skip_ftrs  = self.encoder(x)

        x = self.bottleneck(x)

        x = self.decoder(x, skip_ftrs[::-1])

        x = self.final_expansion(x)

        x = self.head(x.permute(0,3,1,2))

        return x
    
    
### Model training
def train_epoch(model, dataloader):
    model.train()
    losses= []
    for x, y in dataloader:
        optimizer.zero_grad()
        out = model.forward(x.to(DEVICE)) #PyTorchのmodelは、init関数とforword関数を持ちます。
        loss = loss_fn(out, y.to(DEVICE)).to(DEVICE)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def validate_epoch(model, dataloader):
    model.eval() #Sets the module in evaluation mode.
    losses = []
    with torch.no_grad():
        for x, y in dataloader:
            out = model.forward(x.to(DEVICE))
            loss = loss_fn(out, y.to(DEVICE)).to(DEVICE)
            losses.append(loss.item())
    return np.mean(losses)

def train(model, epochs, min_epochs, early_stop_count):

    best_valid_loss = float('inf')
    EARLY_STOP = early_stop_count
    for ep in range(epochs):
        train_loss = train_epoch(model, train_loader)
        valid_loss = validate_epoch(model, valid_loader)

        print(f'Epoch: {ep}: train_loss={train_loss:.5f}, valid_loss={valid_loss:.5f}')
        if ep>min_epochs:
            if(valid_loss < best_valid_loss):
                best_valid_loss = valid_loss
                EARLY_STOP = early_stop_count
            else:
                # EARLY_STOP -= 1
                EARLY_STOP = early_stop_count  #not stop
                if EARLY_STOP <= 0:
                    return train_loss, valid_loss
    return train_loss, valid_loss

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = device
numclss = 1 #1000 in ori code
model = SwinUNet(size,size,3,BATCH_SIZE,numclss,3,4).to(DEVICE) #(H, W, ch, C, batch, num_class?, num_blocks=3, patch_size = 4) #ori: 224,224,1,32,1,3,4
## ?? num_class=2 is error (target and input size different error, but num_class=1 can work)

for p in model.parameters():
    if p.dim() > 1:
            nn.init.kaiming_uniform_(p)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.BCEWithLogitsLoss()


### Training  ## Need visualization of loss
## Original Medium
# train(model, epochs=100, min_epochs=25, early_stop_count=5) #



""" borrow from UNet """
#IoUのクラスを定義
from sklearn.metrics import confusion_matrix
def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.to('cpu').detach().numpy()
    y_true = y_true.to('cpu').detach().numpy()
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # compute mean iou
    intersection = np.diag(current) #extract diagonal elements
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)


class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        # intersection = (inputs * targets).sum()
        # total = (inputs + targets).sum()
        # union = total - intersection

        # IoU = (intersection + smooth)/(union + smooth)
        
        inputs = nn.Sigmoid()(inputs)
        inputs[inputs<0.025] = 0 #set threshold
        inputs[inputs!=0] = 1
        targets[targets<0.039] =0
        targets[targets!=0] =1
        
        IoU = compute_iou(inputs, targets)

        return IoU

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)
    
### Training save check point

accuracy_metric = IoU()
num_epochs=100#100
valid_loss_min = np.Inf

# checkpoint_path = os.path.join(model_dir, 'chkpoint_Medc3_Bi01.pth') #chkpoint_
# best_model_path = os.path.join(model_dir, 'model_Medc3_Bi01.pth') #bestmodel.pt
checkpoint_path = os.path.join(model_dir, 'chkpoint_Medc3_Bi_SDG.pth') #chkpoint_
best_model_path = os.path.join(model_dir, 'model_Medc3_Bi_SDG.pth') #bestmodel.pt


# early_stop_count = 5
# min_epochs = 25
best_valid_loss = float('inf')
# EARLY_STOP = early_stop_count

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
    model.train() #added
    for x, y in pbar:
      x_train = torch.autograd.Variable(x).to(device) #UNet
      y_train = torch.autograd.Variable(y).to(device) #UNet
      optimizer.zero_grad()
      out = model.forward(x.to(DEVICE))
      loss = loss_fn(out, y.to(DEVICE)).to(DEVICE)
      train_loss.append(loss.item())
      
      ## 精度評価
      score = accuracy_metric(out, y) #y_train
      loss.backward()
      optimizer.step()
      train_score.append(score.item())
      pbar.set_description(f"Epoch: {epoch+1}, loss: {losses_value}, IoU: {score}")
    
    #<------------Validation------------------------->
    with torch.no_grad():
      for image,mask in valid_loader:
        model.eval()
        
        out_val = model.forward(image.to(DEVICE))
        loss_val = loss_fn(out_val, mask.to(DEVICE)).to(DEVICE)
        valid_loss.append(loss_val.item())
        
        ## 精度評価
        score = accuracy_metric(out_val, mask.to(DEVICE))
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
        'optimizer': optimizer.state_dict(),
        'valid_score': total_valid_score[-1]
    }
    torch.save(checkpoint, checkpoint_path)

    # 評価データにおいて最高精度のモデルのcheckpointの保存
    if total_valid_loss[-1] <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,total_valid_loss[-1]))
        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = total_valid_loss[-1]

    print("")

""" """

# Plot model loss and IoU
plt.figure(1)
plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.plot(range(1,num_epochs+1), total_train_loss, label="Train Loss")
plt.plot(range(1,num_epochs+1), total_valid_loss, label="Valid Loss")
plt.title("Loss")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range(1,num_epochs+1), total_train_score, label="Train Score")
plt.plot(range(1,num_epochs+1), total_valid_score, label="Valid Score")
plt.title("Score (IoU)")
plt.xlabel("epochs")
plt.ylabel("IoU")
plt.legend()
plt.savefig(model_dir+os.sep+f"loss_Medc3_test_{total_valid_score[-1]}.png") #
plt.show()

### Visualizing results
fig, ax = plt.subplots(3,4, figsize=(10,8))
with torch.no_grad():
    for i in range(3):
        x_og,y_og = next(iter(train_loader)) #next and inter method employs values one by one
        x = x_og[0]
        x = np.transpose(x, (1, 2, 0)) #for imshow
        y = y_og[0]
        ax[i,0].imshow(x.squeeze(0).squeeze(0)) #cmap='gray'
        ax[i,0].set_title('Image')
        ax[i,1].imshow(y.squeeze(0).squeeze(0)) #, cmap='gray'
        # y1 = y.squeeze(0).squeeze(0)
        # y1[y1<0.039] =0
        # ax[i,1].imshow(y1) #, cmap='gray'
        ax[i,1].set_title('Mask')
        x_og = x_og.to(DEVICE)
        out = model(x_og[:1]) #x_og[:1] tensor #maybe this is for the prediction
        out = nn.Sigmoid()(out)
        # out = nn.Softmax()(out) #can't...
        out = out.squeeze(0).squeeze(0).cpu()
        ax[i,2].imshow(out) #cmap='gray'
        ax[i,2].set_title('Prediction')
        ax[i,3].imshow((out>0.025).float(), cmap='gray')
        ax[i,3].set_title('Threshold Prediction')
plt.show()

### Visualizing results for paper
## Collect imgs from tif
img_dir = f'/Volumes/PortableSSD 1/Malaysia/01_Blueprint/SDGuthrie/03_UNet/_retraining/1_training_dataset/img'
mask_dir = '/Volumes/PortableSSD 1/Malaysia/01_Blueprint/SDGuthrie/03_UNet/_retraining/1_training_dataset/annoBi01'
out_dir = '/Volumes/PortableSSD 1/Malaysia/01_Blueprint/SDGuthrie/03_UNet/2_out05m_retraining/for_paper'
images = glob(img_dir + os.sep + "*.tif")
masks = glob(mask_dir + os.sep + "*.tif")
outs = glob(out_dir + os.sep + "*.tif")

ids = [os.path.basename(i)[:-4] for i in images]
ids_sample = random.sample(ids, 3)

images_select = [i for i in images if os.path.basename(i)[:-4] in ids_sample]
mask_select = [i for i in masks if os.path.basename(i)[:-4] in ids_sample]
out_select = [i for i in outs if os.path.basename(i)[:-4] in ids_sample]

fig, ax = plt.subplots(3, 3, figsize=(12, 12))
for i in range(len(images_select)):
    # Read image, mask, and prediction
    img = io.imread(images_select[i]).astype('float32')  # RGB image
    mask = io.imread(mask_select[i]).astype('float32')  # Single-channel mask
    pred = io.imread(out_select[i]).astype('float32')  # Prediction (RGB)

    # Plot in subplots
    ax[i, 0].imshow(img / 255.0)  # Normalize RGB image for display
    ax[i, 0].set_title("Image")
    ax[i, 0].axis("off")
    
    ax[i, 1].imshow(mask, cmap="gray")  # Plot mask as grayscale
    ax[i, 1].set_title("Mask")
    ax[i, 1].axis("off")
    
    ax[i, 2].imshow((pred>0.025), cmap='Oranges_r') # Normalize prediction (assumed RGB)
    ax[i, 2].set_title("Prediction")
    ax[i, 2].axis("off")

# Save and show the plot
output_file = os.path.join(out_dir, "prediction_sample.png")
plt.savefig(output_file, bbox_inches='tight')
plt.show()




""" """
""" Prediction """
### Load model test
# best_model_path = '/Volumes/PortableSSD 1/MAlaysia/01_Blueprint/Pegah_san/03_UNet/2_retraining/1_training_dataset/model/model_Medc3.pth'
# model_load = torch.load(best_model_path)

# output_dir = '/Volumes/PortableSSD/Malaysia/01_Blueprint/Pegah_san/03_UNet/OUT_swinBi'
output_dir = '/Volumes/PortableSSD/Malaysia/01_Blueprint/SDGuthrie/03_UNet/2_out05m_retraining/extent3'

### Converting without tfw
# input_dir = '/Volumes/PortableSSD/Malaysia/01_Blueprint/Pegah_san/03_UNet/1_tiles/images'
# input_dir2 = '/Volumes/PortableSSD/Malaysia/01_Blueprint/Pegah_san/03_UNet/1_tiles/imagesTiff'
input_dir2 = '/Volumes/PortableSSD/Malaysia/01_Blueprint/SDGuthrie/03_UNet/1_tiles05m/extent3/imagesTiff'

file_list =glob(input_dir2+ os.sep +"*.tif")

# for input_path2 in file_list:
#     with rasterio.open(input_path2) as src:
#         arr = src.read()
#         profile = src.profile
#         transform =  src.transform
#         filename = os.path.basename(input_path2)
#         out_path2 = input_dir2 + os.sep + filename
#         with rasterio.open(out_path2, 'w', **profile) as dst:
#           dst.write(arr)
    


### Fail DataLoader がtfw拾ってしまう
def get_predic_transform(size):
    return A.Compose(
        [
          A.Resize(size, size),
          #正規化(こちらの細かい値はalbumentations.augmentations.transforms.Normalizeのデフォルトの値を適用)
          A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

          ToTensorV2()
          ])

class LoadPredicDataSet(Dataset):
        def __init__(self,image_folder, transform=None):
            self.image_folder = image_folder
            self.transforms = get_predic_transform(size) #Augumentationで定義した

        # __len__: 組み込み関数？ここでは独自に定義？　https://blog.codecamp.jp/python-class-code
        def __len__(self):
            self.image_list =glob.glob(self.image_folder+ "/*.tif")
            return len(self.image_list)


        def __getitem__(self,idx): #idx: indexじゃないとだめぽい
            image_folder = self.image_folder
            mask_folder = image_folder #dammy
            image_path = glob.glob(input_dir2+ "/*.tif")#[idx]
            # image_path = '/Volumes/PortableSSD 1/MAlaysia/01_Blueprint/Pegah_san/03_UNet/1_tiles/images/000000000024.tif'
            # image_path = "/Volumes/PortableSSD 1/MAlaysia/01_Blueprint/Pegah_san/03_UNet/2_retraining/1_training_dataset/img/8.tif"

            #画像データの取得
            # img = io.imread(image_path)[:,:,:3].astype('float32') #(64,64,3)
            img = Image.open(input_path).convert('RGB') 
            # img = transform.resize(img,(size,size)) #Affine error??
            img = np.array(img).astype('float32')
            img = np.resize(img, (size,size,3))

            mask = self.get_mask(mask_folder, size, size, idx).astype('float32')

            augmented = self.transforms(image=img) #mask=mask #is_check_shapes=False
            img = augmented['image']
            # mask = augmented['mask']
            # mask = mask[0].permute(2, 0, 1)
            # mask = mask.permute(2, 0, 1)
            # return (img,mask)
            return img
        
        #マスクデータの取得
        def get_mask(self,mask_folder,IMG_HEIGHT, IMG_WIDTH, idx):
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1)) #dtype=np.bool
            mask_ = os.path.join(self.image_folder,natsorted(os.listdir(mask_folder))[idx]) #dammy
            # mask_ = io.imread(os.path.join(mask_folder,mask_))
            mask_ = Image.open(os.path.join(mask_folder,mask_))
            # mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH)) #(256, 256)にする
            mask_ = np.array(img).astype('float32')
            mask_ = np.resize(img, (1, size,size))
            mask_ = np.expand_dims(mask_,axis=-1) #np.expand_dims指定位置に次元追加（）shapeが変わる
            mask = np.maximum(mask, mask_)

            return mask
        
# #trained model exoecxts input as (Batch_size, 3, H, W) (10,3,256,256)
preprocess = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
])


model = SwinUNet(size,size,3,BATCH_SIZE,1,3,4).to(DEVICE) #seems to need define


### Load model
def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


model, optimizer, start_epoch, valid_loss_min = load_ckp(best_model_path, model, optimizer)


    
### Run test1
for input_path in tqdm(file_list):

  img_org = Image.open(input_path).convert('RGB') 
  img = preprocess(img_org)   # (3, H, W) -> (1, 3, H, W)
  img_batch = img.unsqueeze(0)

  # Prediction
  test_dataloader = DataLoader(img_batch, batch_size= BATCH_SIZE) #,shuffle=True
  
  with torch.no_grad():
      for data in test_dataloader:
        # data = torch.autograd.Variable(data, volatile=True).to(device)
        data = data.to(device)
        o = model(data)
        out = nn.Sigmoid()(o)
        tm=out[0][0].data.cpu().numpy() #arrayになる
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
        # 'dtype': profile["dtype"],  # Data type of the array
        'dtype': "float32",  # Data type of the array
        'crs': profile["crs"],  # Coordinate Reference System (adjust as needed)
        'transform': transform  # Spatial transformation
    }

      with rasterio.open(out_path, 'w', **profile_new) as dst:
        dst.write(tm,1)
        
    
### Run test2  DataLoader がtfw拾ってしまう -> geotif convert in input_dir2
### Sigmoid makes positive values
# with torch.no_grad():
#     predic_dataset = LoadPredicDataSet(input_dir2, transform=get_predic_transform(size)) #better augmentation only train?
#     test_dataloader = DataLoader(dataset=predic_dataset, batch_size=BATCH_SIZE)
    
    
#     x_og = next(iter(test_dataloader)) #next and inter method employs values one by one #error with tfw
#     x = x_og[0]
#     x = np.transpose(x, (1, 2, 0)) #for imshow
#     ax[i,0].imshow(x.squeeze(0).squeeze(0)) #cmap='gray'
#     ax[i,0].set_title('Image')
#     x_og = x_og.to(DEVICE)
#     out = model(x_og[:1]) #x_og[:1] tensor #maybe this is for the prediction
#     out = nn.Sigmoid()(out)
#     out = out.squeeze(0).squeeze(0).cpu()
#     ax[i,2].imshow(out) #cmap='gray'
#     ax[i,2].set_title('Prediction')

#     #check
#      # figure, ax = plt.subplots()
#      # ax.imshow(tm, interpolation="nearest", cmap="gray")
  
#     out_path = output_dir + os.sep + os.path.basename(input_path)
  
#     with rasterio.open(input_path) as src:
#         profile = src.profile
#         transform =  src.transform
  
#         profile_new = {
#           'driver': 'GTiff',  # GeoTIFF format
#           'height': profile["height"],  # Number of rows
#           'width': profile["width"],   # Number of columns
#           'count': 1,  # Number of bands (e.g., 1 for grayscale)
#           # 'dtype': profile["dtype"],  # Data type of the array
#           'dtype': "float32",  # Data type of the array
#           'crs': profile["crs"],  # Coordinate Reference System (adjust as needed)
#           'transform': transform  # Spatial transformation
#       }
  
#         with rasterio.open(out_path, 'w', **profile_new) as dst:
#           dst.write(tm,1)


