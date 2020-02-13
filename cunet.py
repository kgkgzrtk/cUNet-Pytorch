import torch
import torch.nn as nn
from utils import AdaIN, HalfDropout, BatchNorm
from nets import r_double_conv


class Conditional_UNet(nn.Module):

    def init_weight(self, std=0.2):
        for m in self.modules():
            cn = m.__class__.__name__
            if cn.find('Conv') != -1:
                m.weight.data.normal_(0., std)
            elif cn.find('Linear') != -1:
                m.weight.data.normal_(1., std)
                m.bias.data.fill_(0)

    def __init__(self, num_classes):
        super(Conditional_UNet, self).__init__()

        self.dconv_down1 = r_double_conv(3, 64)
        self.dconv_down2 = r_double_conv(64, 128)
        self.dconv_down3 = r_double_conv(128, 256)
        self.dconv_down4 = r_double_conv(256, 512)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.3)
        #self.dropout_half = HalfDropout(p=0.3)
        
        self.adain3 = AdaIN(512, num_classes=num_classes)
        self.adain2 = AdaIN(256, num_classes=num_classes)
        self.adain1 = AdaIN(128, num_classes=num_classes)

        self.dconv_up3 = r_double_conv(256 + 512, 256)
        self.dconv_up2 = r_double_conv(128 + 256, 128)
        self.dconv_up1 = r_double_conv(64 + 128, 64)
        
        self.conv_last = nn.Conv2d(64, 3, 1)
        self.activation = nn.Tanh()
        #self.init_weight() 
        
    def forward(self, x, c):

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)

        #dropout
        #x = self.dropout_half(x)
        
        x = self.adain3(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)

        x = self.adain2(x, c)
        x = self.upsample(x)        
        x = self.dropout(x)
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)

        x = self.adain1(x, c)
        x = self.upsample(x)        
        x = self.dropout(x)
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return self.activation(out)

