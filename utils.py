import os

import torch
import torch.nn as nn

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
        self.emb = nn.Embedding(num_classes, num_classes)

    def c_norm(self, x, bs, ch):
        assert isinstance(x, torch.cuda.FloatTensor)
        x_var = x.var(dim=-1) + self.eps
        x_std = x_var.sqrt().view(bs, ch, 1, 1)
        x_mean = x.mean(dim=-1).view(bs, ch, 1, 1)
        return x_std, x_mean

    def forward(self, x, y):
        assert x.size(0)==y.size(0)
        size = x.size()
        bs, ch = size[:2]
        x_ = x.view(bs, ch, -1)
        y_ = self.l1(y).view(bs, ch, -1)
        x_std, x_mean = self.c_norm(x_, bs, ch)
        y_std, y_mean = self.c_norm(y_, bs, ch)

        out =   ((x - x_mean.expand(size)) / x_std.expand(size)) \
                * y_std.expand(size) + y_mean.expand(size)
        return out

class HalfDropout(nn.Module):
    def __init__(self, p=0.3):
        super(HalfDropout, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        ch = x.size(1)
        a = x[:, :ch//2]
        a = self.dropout(a)
        b = x[:, ch//2:]
        out = torch.cat([a,b], dim=1)
        return out

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )
