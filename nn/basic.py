import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Swish


class EmptyLayer(nn.Module):
    def forward(self, x):
        return x


# norm-swish-conv
class NSC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 groups=1,
                 dilation=1):
        super(NSC, self).__init__()
        self.block = nn.Sequential(
            EmptyLayer() if in_channels % 32 != 0 else nn.GroupNorm(
                32, in_channels),
            Swish(),
            WSConv2d(
                in_channels,
                out_channels,
                ksize,
                stride=stride,
                padding=(ksize - 1) // 2 - 1 + dilation,
                groups=groups,
                dilation=dilation,
                bias=False,
            ),
        )

    def forward(self, x):
        return self.block(x)


class CNS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 groups=1,
                 dilation=1):
        super(CNS, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                ksize,
                stride=stride,
                padding=(ksize - 1) // 2 - 1 + dilation,
                groups=groups,
                dilation=dilation,
            ),
            nn.GroupNorm(32, out_channels) if out_channels %
            32 == 0 else nn.GroupNorm(8, out_channels) if out_channels %
            8 == 0 else EmptyLayer(),
            Swish(),
        )

    def forward(self, x):
        return self.block(x)
