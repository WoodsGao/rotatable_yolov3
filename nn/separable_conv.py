import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 dilation=1,
                 bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                ksize,
                stride,
                (ksize - 1) // 2 - 1 + dilation,
                dilation,
                in_channels,
                bias,
            ),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
        )

    def forward(self, x):
        return self.conv(x)
