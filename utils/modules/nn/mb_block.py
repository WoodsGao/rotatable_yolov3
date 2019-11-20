import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Swish, CNS, SELayer, EmptyLayer, DropConnect


class MbBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 expand_ratio=6,
                 drop_rate=0.2,
                 reps=1):
        super(MbBlock, self).__init__()
        blocks = [
            MbConv(in_channels, out_channels, ksize, stride, expand_ratio,
                   drop_rate)
        ]
        for i in range(reps - 1):
            blocks.append(
                MbConv(out_channels, out_channels, ksize, 1, expand_ratio,
                       drop_rate))
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MbConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 expand_ratio=6,
                 drop_rate=0.2):
        super(MbConv, self).__init__()
        mid_channels = in_channels * expand_ratio
        if in_channels == out_channels and stride == 1:
            self.add = True
        else:
            self.add = False
        self.block = nn.Sequential(
            CNS(in_channels, mid_channels, 1)
            if expand_ratio > 1 else EmptyLayer(),
            CNS(mid_channels,
                mid_channels,
                ksize=ksize,
                stride=stride,
                groups=mid_channels),
            SELayer(mid_channels),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            DropConnect(drop_rate)
            # nn.Dropout(drop_rate)
            if self.add and drop_rate > 0 else EmptyLayer(),
        )

    def forward(self, x):
        if self.add:
            identity = x
        x = self.block(x)
        if self.add:
            x = x + identity
        return x
