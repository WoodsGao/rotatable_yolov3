import torch.nn as nn
from . import CNS, EmptyLayer, SELayer


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 se_block=False):
        super(ResBlock, self).__init__()
        if stride == 1 and in_channels == out_channels:
            self.add = True
        else:
            self.add = False
        self.block = nn.Sequential(
            CNS(in_channels, out_channels // 2, 1),
            CNS(
                out_channels // 2,
                out_channels // 2,
                stride=stride,
                dilation=dilation,
                groups=32 if out_channels % 64 == 0 else 1,
            ),
            # See https://arxiv.org/pdf/1604.04112.pdf
            CNS(out_channels // 2, out_channels, 1, activate=False),
        )

    def forward(self, x):
        f = self.block(x)
        if self.add:
            f += x
        return f
