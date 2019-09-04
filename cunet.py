import torch
import torch.nn as nn
from utils import ConditionalNorm, double_conv


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
        
        self.cbn3 = ConditionalNorm(512, num_classes=num_classes)
        self.cbn2 = ConditionalNorm(256, num_classes=num_classes)
        self.cbn1 = ConditionalNorm(128, num_classes=num_classes)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(64 + 128, 64)
        
        self.conv_last = nn.Conv2d(64, 3, 1)

        
        
    def forward(self, x, c):

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        #embedding vector to spatial vector
        
        x = self.cbn3(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)

        x = self.cbn2(x, c)
        x = self.upsample(x)        
        x = self.dropout(x)
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)

        x = self.cbn1(x, c)
        x = self.upsample(x)        
        x = self.dropout(x)
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

'''
class UNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.emb_dims = num_classes
        self.emb = nn.Embedding(num_classes, self.emb_dims)
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512 + self.emb_dims, 256)
        self.dconv_up2 = double_conv(128 + 256 + self.emb_dims, 128)
        self.dconv_up1 = double_conv(128 + 64 + self.emb_dims, 64)
        
        self.conv_last = nn.Conv2d(64, 3, 1)

        
        
    def forward(self, x, c):

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        emb_c = self.emb(c)
        #embedding vector to spatial vector
        emb_c = emb_c.repeat(1,4)
        emb_c = emb_c.view([-1, self.emb_dims, 2, 2])
        while x.shape[2]!=emb_c.shape[2]:
            emb_c = self.upsample(emb_c)
        x = torch.cat([x, emb_c], dim=1)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)

        while x.shape[2]!=emb_c.shape[2]:
            emb_c = self.upsample(emb_c)
        x = torch.cat([x, emb_c], dim=1)

        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)


        while x.shape[2]!=emb_c.shape[2]:
            emb_c = self.upsample(emb_c)
        x = torch.cat([x, emb_c], dim=1)

        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
'''
