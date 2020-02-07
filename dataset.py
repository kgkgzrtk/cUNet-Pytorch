import os

import pandas as pd
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader

def _collate_fn(batch):
    data = [v[0] for v in batch]
    target = [v[1] for v in batch]
    target = torch.FloatTensor(target)
    return [data, target]

def get_class_id_from_string(string):
    s_li = ['sunny','cloudy', 'rain', 'snow', 'foggy'] 
    if not string in s_li: raise
    else: return s_li.index(string)


class FlickrDataLoader(Dataset):
    def __init__(self, image_root, df, columns, transform=None):
        super(FlickrDataLoader, self).__init__()
        #init
        self.root = image_root
        self.columns = columns
        self.photo_id = df['photo'].to_list()
        df_ = df.loc[:, columns].fillna(0)
        self.conditions = (df_ - df_.mean())/df_.std()
        self.labels = df['condition2']
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
        try:
            image = Image.open(os.path.join(self.root, self.photo_id[idx]+'.jpg'))
        except:
            return self.__getitem__(idx)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.get_condition(idx)
        return image, label

class ImageLoader(Dataset):
    def __init__(self, paths, transform=None):
        super(ImageLoader, self).__init__()
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.paths[idx])
        except:
            return self.__getitem__(idx)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, True

class ClassImageLoader(Dataset):
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


