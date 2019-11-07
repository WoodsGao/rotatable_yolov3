import torch.nn as nn
from . import SELayer, Swish, CReLU


class EmptyLayer(nn.Module):
    def forward(self, x):
        return x


class BLD(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 groups=1,
                 dilation=1,
                 activate=Swish()):
        super(BLD, self).__init__()
        if activate is None:
            activate = EmptyLayer()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels), activate,
            nn.Conv2d(in_channels *
                      2 if isinstance(activate, CReLU) else in_channels,
                      out_channels,
                      ksize,
                      stride=stride,
                      padding=(ksize - 1) // 2 - 1 + dilation,
                      groups=groups,
                      dilation=dilation,
                      bias=False))

    def forward(self, x):
        return self.block(x)


class DBL(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 groups=1,
                 dilation=1,
                 activate=Swish()):
        super(DBL, self).__init__()
        if activate is None:
            activate = EmptyLayer()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels //
                      2 if isinstance(activate, CReLU) else out_channels,
                      ksize,
                      stride=stride,
                      padding=(ksize - 1) // 2 - 1 + dilation,
                      groups=groups,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            activate,
        )

    def forward(self, x):
        return self.block(x)
