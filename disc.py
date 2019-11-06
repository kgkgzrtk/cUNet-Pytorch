import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.nn import utils

from utils import ConditionalNorm

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=1),
        nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2),
        nn.BatchNorm2d(out_channels, affine=False),
        nn.LeakyReLU(0.2, inplace=True)
    )
    
def sn_double_conv(in_channels, out_channels):
    return nn.Sequential(
        utils.spectral_norm(
            nn.Conv2d(in_channels, in_channels, 3, padding=1)),
        utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2)),
        nn.LeakyReLU(0.2, inplace=True)
    )   

class SNDisc(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = sn_double_conv(3, 64)
        self.conv2 = sn_double_conv(64, 128)
        self.conv3 = sn_double_conv(128, 256)
        self.conv4 = sn_double_conv(256, 512)
        [init.xavier_uniform_(
            getattr(self, 'conv{}'.format(i))[j].weight,
            np.sqrt(2)
            ) for i in range(1, 5) for j in range(2)]

        self.l = utils.spectral_norm(nn.Linear(512, 1))
        init.xavier_uniform_(self.l.weight)

        self.embed = utils.spectral_norm(nn.Embedding(num_classes, 512))
        init.xavier_uniform_(self.embed.weight)
        
    def forward(self, x, c=None):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        x = torch.sum(c4, [2,3]) # global pool
        out = self.l(x)
        e_c = self.embed(c)
        if c is not None:
            out += torch.sum(e_c * x, dim=1, keepdim=True)
        return [out, c1, c2, c3, c4]

class CBNDisc(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        self.cbn1 = ConditionalNorm(64, num_classes)
        self.cbn2 = ConditionalNorm(128, num_classes)
        self.cbn3 = ConditionalNorm(256, num_classes)
        self.conv1 = double_conv(3, 64)
        self.conv2 = double_conv(64, 128)
        self.conv3 = double_conv(128, 256)
        self.conv4 = double_conv(256, 512)
        self.l = nn.Linear(512*16*16, 1)
        
    def forward(self, x, c):
        x = self.conv1(x)
        x = self.cbn1(x, c)
        x = self.conv2(x)
        x = self.cbn2(x, c)
        x = self.conv3(x)
        x = self.cbn3(x, c)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)


        out = self.l(x)
        return out
