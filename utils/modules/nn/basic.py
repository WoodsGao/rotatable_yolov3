import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Mish, WSConv2d, EmptyLayer, AdaGroupNorm


class CNS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 groups=1,
                 dilation=1,
                 activate=True,
                 inplace=False):
        super(CNS, self).__init__()
        # conv = WSConv2d if groups == 1 else nn.Conv2d
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      ksize,
                      stride=stride,
                      padding=(ksize - 1) // 2 * dilation,
                      groups=groups,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            Mish() if activate else EmptyLayer(),
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
        self.block = nn.Sequential(
            CNS(in_channels,
                in_channels,
                ksize,
                stride=stride,
                groups=in_channels,
                dilation=dilation,
                inplace=True),
            CNS(in_channels, out_channels, 1, activate=activate, inplace=True),
        )

    def forward(self, x):
        return self.block(x)
