import torch
import torch.nn as nn
from . import NSC, EmptyLayer


class CutLayer(nn.Module):
    def __init__(self, cut_idx):
        super(CutLayer, self).__init__()
        self.cut_idx = cut_idx

    def forward(self, x):
        return x[:, :self.cut_idx]


class DenseBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 se_block=False):
        super(DenseBlock, self).__init__()
        assert in_channels == out_channels or 2 * in_channels == out_channels
        assert in_channels % 2 == 0
        assert stride == 1 or stride == 2
        out_channels = out_channels // 2
        if stride == 1:
            self.downsample = CutLayer(out_channels)
        else:
            self.downsample = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.block = nn.Sequential(
            NSC(in_channels, out_channels // 2, 1),
            NSC(
                out_channels // 2,
                out_channels,
                stride=stride,
                dilation=dilation,
                groups=32 if out_channels % 64 == 0 else 1,
            ),
        )

    def forward(self, x):
        x = torch.cat([self.downsample(x), self.block(x)], 1)
        return x
