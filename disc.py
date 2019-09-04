import torch
import torch.nn as nn
from utils import ConditionalNorm

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=1),
        nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2),
        nn.BatchNorm2d(out_channels, affine=False),
        nn.LeakyReLU(0.2, inplace=True)
    )   

class Discriminator(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        self.cbn1 = ConditionalNorm(64)
        self.cbn2 = ConditionalNorm(128)
        self.cbn3 = ConditionalNorm(256)
        self.conv1 = double_conv(3, 64)
        self.conv2 = double_conv(64, 128)
        self.conv3 = double_conv(128, 256)
        self.conv4 = double_conv(256, 512)
        self.embed = nn.Linear(num_classes, 512)
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
