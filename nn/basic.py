import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Swish, SeparableConv2d, WSConv2d, EmptyLayer, AdaGroupNorm


# conv norm swish
class CNS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 groups=1,
                 dilation=1,
                 activate=True):
        super(CNS, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              ksize,
                              stride=stride,
                              padding=(ksize - 1) // 2 * dilation,
                              groups=groups,
                              dilation=dilation,
                              bias=False)
        self.norm = AdaGroupNorm(out_channels)
        if activate:
            self.activation = Swish()
        else:
            self.activation = EmptyLayer()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


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
        self.norm = AdaGroupNorm(out_channels)
        if activate:
            self.activation = Swish()
        else:
            self.activation = EmptyLayer()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
