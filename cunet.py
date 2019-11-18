import torch
import torch.nn as nn
from utils import AdaIN, double_conv


class Conditional_UNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.dropout = nn.Dropout(p=0.3)
        
        self.adain3 = AdaIN(512, num_classes=num_classes)
        self.adain2 = AdaIN(256, num_classes=num_classes)
        self.adain1 = AdaIN(128, num_classes=num_classes)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(64 + 128, 64)
        
        self.conv_last = nn.Conv2d(64, 3, 1)
        self.activation = nn.Tanh()

        
        
    def forward(self, x, c, c_=None):

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        #embedding vector to spatial vector
        
        x = self.adain3(x, c, c_)
        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)

        x = self.adain2(x, c, c_)
        x = self.upsample(x)        
        x = self.dropout(x)
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)

        x = self.adain1(x, c, c_)
        x = self.upsample(x)        
        x = self.dropout(x)
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return self.activation(out)

