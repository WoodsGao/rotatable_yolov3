import torch.nn as nn
from . import BLD, EmptyLayer


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 se_block=False):
        super(ResBlock, self).__init__()
        if stride == 1 and in_channels == out_channels:
            self.downsample = EmptyLayer()
        else:
            self.downsample = nn.Conv2d(in_channels, out_channels, 3, stride,
                                        1)
        self.block = nn.Sequential(
            BLD(in_channels, out_channels // 2, 1),
            BLD(
                out_channels // 2,
                out_channels,
                dilation=dilation,
                groups=32 if out_channels % 64 == 0 else 1,
            ),
        )

    def forward(self, x):
        return self.downsample(x) + self.block(x)
