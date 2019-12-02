from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Swish, CNS, SELayer, EmptyLayer


class MbBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 dilation=1,
                 expand_ratio=6,
                 drop_rate=0.2,
                 reps=1):
        super(MbBlock, self).__init__()
        blocks = []
        for i in range(reps):
            blocks.append(
                MbConv(in_channels if i == 0 else out_channels, out_channels,
                       ksize, stride if i == 0 else 1, dilation, expand_ratio,
                       drop_rate))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class MbConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 dilation=1,
                 expand_ratio=6,
                 drop_rate=0.2):
        super(MbConv, self).__init__()
        self.drop_rate = drop_rate
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
                groups=mid_channels,
                dilation=dilation),
            SELayer(mid_channels),
            # See https://arxiv.org/pdf/1604.04112.pdf
            CNS(mid_channels, out_channels, 1, activate=False),
        )

    def forward(self, x):
        f = self.block(x)
        if self.add:
            if self.training:
                if random() > self.drop_rate:
                    f.add_(x)
            else:
                f.add_(x.mul(1 - self.drop_rate))
        return f
