import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Swish, CNS, SELayer, EmptyLayer


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
            CNS(mid_channels, out_channels, 1),
        )

    def forward(self, x):
        if self.add:
            identity = x
        x = self.block(x)
        if self.add:
            x = x + identity
        return x
