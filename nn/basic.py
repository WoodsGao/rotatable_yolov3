import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Swish


class EmptyLayer(nn.Module):
    def forward(self, x):
        return x

# conv norm swish
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
            nn.BatchNorm2d(out_channels),
            Swish(),
        )

    def forward(self, x):
        return self.block(x)
