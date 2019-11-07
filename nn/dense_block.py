import torch
import torch.nn as nn
from . import DBL, EmptyLayer


class DenseBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 drop_rate=0.5,
                 se_block=False):
        super(DenseBlock, self).__init__()
        assert in_channels == out_channels or 2 * in_channels == out_channels
        assert in_channels % 2 == 0
        assert stride == 1
        self.out_channels = out_channels // 2
        self.block = nn.Sequential(
            DBL(in_channels, out_channels // 4, 1),
            DBL(
                out_channels // 4,
                out_channels // 4,
                stride=stride,
                dilation=dilation,
                groups=out_channels // 4,
            ),
            DBL(
                out_channels // 4,
                out_channels // 2,
                1,
            ),
            nn.Dropout(drop_rate) if drop_rate > 0 else EmptyLayer(),
        )

    def forward(self, x):
        x = torch.cat([x[:, :self.channels], self.block(x)], 1)
        return x
