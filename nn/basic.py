import torch.nn as nn
from . import Swish


class EmptyLayer(nn.Module):
    def forward(self, x):
        return x


# norm-swish-conv
class NSC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 groups=1,
                 dilation=1):
        super(NSC, self).__init__()
        self.block = nn.Sequential(
            EmptyLayer() if in_channels % 32 != 0 else nn.GroupNorm(
                32, in_channels),
            Swish(),
            nn.Conv2d(
                in_channels,
                out_channels,
                ksize,
                stride=stride,
                padding=(ksize - 1) // 2 - 1 + dilation,
                groups=groups,
                dilation=dilation,
                bias=False,
            ),
        )

    def forward(self, x):
        return self.block(x)


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
<<<<<<< HEAD
            nn.Conv2d(
                in_channels,
                out_channels,
                ksize,
                stride=stride,
                padding=(ksize - 1) // 2 - 1 + dilation,
                groups=groups,
                dilation=dilation,
            ),
            EmptyLayer() if out_channels % 32 != 0 else nn.GroupNorm(
                32, out_channels),
            nn.GroupNorm(32, out_channels),
            Swish(),
=======
            nn.Conv2d(in_channels,
                      out_channels //
                      2 if isinstance(activate, CReLU) else out_channels,
                      ksize,
                      stride=stride,
                      padding=(ksize - 1) // 2 - 1 + dilation,
                      groups=groups,
                      dilation=dilation,
                      bias=False),
            EmptyLayer() if out_channels % 16 != 0 else
            nn.GroupNorm(16, out_channels),
            activate,
>>>>>>> 4118242ff37571bef423c6bb9bfc7ed8f362f15f
        )

    def forward(self, x):
        return self.block(x)
