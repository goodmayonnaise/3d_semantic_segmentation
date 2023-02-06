# -*- coding: utf-8 -*-

from __future__ import print_function

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import cv2 
from einops import rearrange
import os

import torch
from torch.utils.data import Dataset
from torchvision import utils


root_dir   = "/mnt/team_gh/cityscapes"
train_file = os.path.join(root_dir, "train.csv")
val_file   = os.path.join(root_dir, "val.csv")

num_class = 20
means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR

class CityScapesDataset(Dataset):
    def __init__(self, csv_file, input_shape, n_class=num_class):
        self.data = pd.read_csv(csv_file)
        self.n_class   = n_class
        self.input_shape = input_shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name   = self.data.iloc[idx, 0]
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        img = np.array(img)
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        img = torch.FloatTensor(img)
        img = rearrange(img, 'h w c -> c h w')

        label_name = self.data.iloc[idx, 1]
        label = cv2.imread(label_name) 
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)        
        label = cv2.resize(label, (self.input_shape[1], self.input_shape[0]))
        label = np.array(label)
        label = torch.FloatTensor(label)

        # create one-hot encoding
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class): 
            target[c] = (label==c).type(torch.int32).clone().detach()

        sample = {'X': img, 'Y': target, 'l': label}  

        return sample

def show_batch(batch):
    img_batch = batch['X']
    img_batch[:,0,...].add_(means[0])
    img_batch[:,1,...].add_(means[1])
    img_batch[:,2,...].add_(means[2])
    batch_size = len(img_batch)

    grid = utils.make_grid(img_batch)
    plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))

    plt.title('Batch from dataloader')

## Transform class 구현
class ToTensor(object):
    def __call__(self, data):
        print(data)
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

# if __name__ == "__main__":
#     train_data = CityScapesDataset(csv_file=train_file, phase='train')

#     # show a batch
#     batch_size = 4
#     for i in range(batch_size):
#         sample = train_data[i]
#         print(i, sample['X'].size(), sample['Y'].size())

#     dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)

#     for i, batch in enumerate(dataloader):
#         print(i, batch['X'].size(), batch['Y'].size())
    
#         # observe 4th batch
#         if i == 3:
#             plt.figure()
#             show_batch(batch)
#             plt.axis('off')
#             plt.ioff()
#             plt.show()
#             break

# 사용 안함 -------------------------------------------------------------------------
# class Normalization(object): # 주석 풀면 우리 데이터에 맞는 거 
#     def __init__(self, mean=0.5, std=0.5):
#         self.mean = mean
#         self.std = std

#     def __call__(self, data):
#         target, input, label = data['Y'], data['X'], data['l']
#         # img, target, label = data['X'], data['Y'], data['l']

#         input = (input - self.mean) / self.std

#         data = {'label': label, 'input': input}
#         # data = {'X':input, 'Y': target, 'l' : label}
#         return data

# class RandomFlip(object): # 수정 전 
#     def __call__(self, data):
#         label, input = data['label'], data['input']

#         if np.random.rand() > 0.5:
#             label = np.fliplr(label)
#             input = np.fliplr(input)

#         if np.random.rand() > 0.5:
#             label = np.flipud(label)
#             input = np.flipud(input)

#         data = {'label': label, 'input': input}

#         return 
# ---------------------------------------------------------------------------------
