import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Swish, SeparableConv2d, WSConv2d, EmptyLayer, AdaGroupNorm


class CNS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 groups=1,
                 dilation=1):
        super(CNS, self).__init__()
        conv = WSConv2d if groups == 1 else nn.Conv2d
        self.block = nn.Sequential(
            conv(in_channels,
                 out_channels,
                 ksize,
                 stride=stride,
                 padding=(ksize - 1) // 2 - 1 + dilation,
                 groups=groups,
                 dilation=dilation,
                 bias=False),
            nn.GroupNorm(8, out_channels),
            Swish(),
        )

    def forward(self, x):
        return self.block(x)


# conv norm swish
class SeparableCNS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 dilation=1,
                 activate=True):
        super(SeparableCNS, self).__init__()
        self.conv = SeparableConv2d(in_channels,
                                    out_channels,
                                    ksize,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=False)
        self.norm = nn.GroupNorm(8, out_channels)
        if activate:
            self.activation = Swish()
        else:
            self.activation = EmptyLayer()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
