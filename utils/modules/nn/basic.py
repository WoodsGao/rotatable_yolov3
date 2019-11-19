import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Swish


class EmptyLayer(nn.Module):
    def forward(self, x):
        return x


class WSConv2d(nn.Conv2d):
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True)
        weight_mean = weight_mean.mean(dim=2, keepdim=True)
        weight_mean = weight_mean.mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(
            torch.var(weight.view(weight.size(0), -1), dim=1, unbiased=False) +
            1e-12).view(-1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


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
            WSConv2d(
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
            WSConv2d(
                in_channels,
                out_channels,
                ksize,
                stride=stride,
                padding=(ksize - 1) // 2 - 1 + dilation,
                groups=groups,
                dilation=dilation,
            ) if groups == 1 else nn.Conv2d(
                in_channels,
                out_channels,
                ksize,
                stride=stride,
                padding=(ksize - 1) // 2 - 1 + dilation,
                groups=groups,
                dilation=dilation,
            ),
            nn.BatchNorm2d(out_channels) if out_channels % 32 != 0 or groups > 1 else
            nn.GroupNorm(32, out_channels),
            Swish(),
        )

    def forward(self, x):
        return self.block(x)
