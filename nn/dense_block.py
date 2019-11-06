import torch
import torch.nn as nn
from . import BLD, EmptyLayer


class DenseBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 drop_rate=0.5,
                 se_block=True):
        super(DenseBlock, self).__init__()
        assert in_channels == out_channels
        assert in_channels % 2 == 0
        assert stride == 1
        channels = in_channels
        self.channels = channels // 2
        self.block = nn.Sequential(
            BLD(channels, channels // 4, 1),
            BLD(
                channels // 4,
                channels // 4,
                stride=stride,
                dilation=dilation,
                groups=channels // 4,
            ),
            BLD(channels // 4,
                channels // 2,
                1,
                activate=None,
                se_block=se_block),
            nn.Dropout(drop_rate) if drop_rate > 0 else EmptyLayer(),
        )

    def forward(self, x):
        x = torch.cat([x[:, :self.channels], self.block(x)], 1)
        return x
