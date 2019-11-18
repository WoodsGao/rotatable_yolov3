import torch
import torch.nn as nn
import torch.nn.functional as F
from . import CNS, NSC


class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AsppPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 CNS(in_channels, out_channels))

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class Aspp(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(Aspp, self).__init__()
        blocks = []
        blocks.append(CNS(in_channels, out_channels, 1))
        for rate in atrous_rates:
            blocks.append(CNS(in_channels, out_channels, dilation=rate))
        blocks.append(AsppPooling(in_channels, out_channels))
        self.blocks = nn.ModuleList(blocks)
        self.project = nn.Sequential(
            CNS(out_channels * len(blocks), out_channels, 1))

    def forward(self, x):
        outputs = []
        for block in self.blocks:
            outputs.append(block(x))
        x = torch.cat(outputs, 1)
        x = self.project(x)
        return x
