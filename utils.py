import os

import pandas as pd
import torch
import torch.nn as nn

import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader

def _collate_fn(batch):
    data = [v[0] for v in batch]
    target = [v[1] for v in batch]
    target = torch.LongTensor(target)
    return [data, target]

def get_class_id_from_string(string):
    s_li = ['sunny','cloudy', 'rain', 'snow', 'foggy'] 
    if not string in s_li: raise
    else: return s_li.index(string)

class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.bn = nn.BatchNorm2d(in_channel, affine=False)
        self.embed = nn.Embedding(num_classes, in_channel*2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, class_id):
        out = self.bn(input)
        embed = self.embed(class_id)
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta
        return out

class AdaIN(nn.Module):
    def __init__(self, in_channel, num_classes, eps=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.eps= eps
        #bias is good :)
        self.l1 = nn.Linear(num_classes, in_channel*4, bias=True)
        self.l2 = nn.Linear(num_classes, in_channel*4, bias=True)
        self.emb = nn.Embedding(num_classes, num_classes)

    def c_norm(self, x, bs, ch):
        assert isinstance(x, torch.cuda.FloatTensor)
        x_var = x.var(dim=-1) + self.eps
        x_std = x_var.sqrt().view(bs, ch, 1, 1)
        x_mean = x.mean(dim=-1).view(bs, ch, 1, 1)
        return x_std, x_mean

    def forward(self, x, y, z=None):
        assert x.size(0)==y.size(0)
        size = x.size()
        bs, ch = size[:2]
        x_ = x.view(bs, ch, -1)
        if len(y.size())==1:
            y = self.emb(y)
        y_ = self.l1(y).view(bs, ch, -1)
        x_std, x_mean = self.c_norm(x_, bs, ch)
        y_std, y_mean = self.c_norm(y_, bs, ch)

        if z is not None:
            assert x.size(0)==z.size(0)
            if len(z.size())==1:
                z = self.emb(z)
            z_ = self.l1(z).view(bs, ch, -1)
            z_std, z_mean = self.c_norm(z_, bs, ch)
            out =   ((x - z_mean.expand(size)) / z_std.expand(size)) \
                    * y_std.expand(size) + y_mean.expand(size)
        else:
            out =   ((x - x_mean.expand(size)) / x_std.expand(size)) \
                    * y_std.expand(size) + y_mean.expand(size)
        return out

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class FlickrDataLoader(Dataset):
    def __init__(self, image_root, df, columns, transform=None):
        #init
        self.root = image_root
        self.columns = columns
        self.photo_id = df['photo'].to_list()
        df_ = df.loc[:, columns].fillna(0)
        self.conditions = (df_ - df_.mean())/df_.std()
        self.labels = df['condition']
        self.num_classes = len(columns)
        self.transform = transform
        del df, df_

    def __len__(self):
        return len(self.photo_id)
    
    def get_class(self, idx):
        string = self.labels.iloc[idx]
        id = list(self.labels.unique()).index(string)
        del string
        return id

    def get_condition(self, idx):
        c = self.conditions.iloc[idx].to_list()
        c_tensor = torch.from_numpy(np.array(c)).float()
        del c
        return c_tensor

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root, self.photo_id[idx]+'.jpg'))
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        target = self.get_condition(idx)
        return image, target
    

class ImageLoader(Dataset):
    def __init__(self, paths, transform=None):
        #without z-other
        paths = [p for p in paths if 'z-other' not in p]
        #count dirs on root
        path = os.path.commonpath(paths)
        files = os.listdir(path)
        files_dir = [f for f in files if os.path.isdir(os.path.join(path, f)) if 'z-other' not in f]
        #init
        self.paths = paths
        self.classes = files_dir
        self.num_classes = len(files_dir)
        self.transform = transform
    
    def get_class(self, idx):
        string = self.paths[idx].split('/')[-2]
        return get_class_id_from_string(string)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx])
        image = image.convert('RGB')
        target = self.get_class(idx)
        if self.transform:
            image = self.transform(image)
        return image, target

class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, loader=default_loader):
        super(ImageFolder, self).__init__(root,
                transform=transform,
                extensions='jpg'
            )
        
    def __getitem__(self, ind):
        path, target = self.samples[ind]
        image = Image.open(path)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, target
