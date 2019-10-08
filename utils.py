import os

from PIL import Image
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader

def _collate_fn(batch):
    data = [v[0] for v in batch]
    target = [v[1] for v in batch]
    target = torch.LongTensor(target)
    return [data, target]

def get_class_id_from_string(string):
    s_li = ['sunny','cloudy', 'rain', 'snow', 'foggy', 'z-other']
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

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

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
