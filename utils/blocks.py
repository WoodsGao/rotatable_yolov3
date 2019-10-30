import torch
import torch.nn as nn
import math
import torch.nn.functional as F

relu = nn.LeakyReLU(0.1, inplace=True)
bn = nn.BatchNorm2d


class SELayer(nn.Module):
    def __init__(self, filters):
        super(SELayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Sequential(
            nn.Conv2d(filters, filters // 16 if filters >= 32 else 8, 1),
            relu,
            nn.Conv2d(filters // 16 if filters >= 32 else 8, filters, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gap = self.gap(x)
        weight = self.weight(gap)
        return x * weight


class EmptyLayer(nn.Module):
    def forward(self, x):
        return x


class AsppPooling(nn.Module):
    def __init__(self, in_features, out_features):
        super(AsppPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 BLD(in_features, out_features, 1))

    def forward(self, x):
        size = x.size()[2:]
        x = self.gap(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


class Aspp(nn.Module):
    def __init__(self, in_features, out_features, rates=[6, 12, 18]):
        super(Aspp, self).__init__()
        conv_list = [BLD(in_features, out_features, 1)]
        conv_list.append(AsppPooling(in_features, out_features))
        for rate in rates:
            conv_list.append(BLD(in_features, out_features, dilation=rate))
        self.conv_list = nn.ModuleList(conv_list)
        self.project = BLD((2 + len(rates)) * out_features, out_features, 1)

    def forward(self, x):
        outputs = []
        for conv in self.conv_list:
            outputs.append(conv(x))
        outputs = torch.cat(outputs, 1)
        outputs = self.project(outputs)
        return outputs


class BLD(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 ksize=3,
                 stride=1,
                 groups=1,
                 dilation=1,
                 activate=True,
                 se_block=False):
        super(BLD, self).__init__()
        if activate:
            activate = relu
        else:
            activate = EmptyLayer()
        if se_block:
            se_block = SELayer(in_features)
        else:
            se_block = EmptyLayer()
        self.bld = nn.Sequential(
            bn(in_features), se_block, activate,
            nn.Conv2d(in_features,
                      out_features,
                      ksize,
                      stride=stride,
                      padding=(ksize - 1) // 2 - 1 + dilation,
                      groups=groups,
                      dilation=dilation,
                      bias=False))

    def forward(self, x):
        return self.bld(x)


class ResBlock(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 stride=1,
                 dilation=1,
                 se_block=True):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            BLD(in_features, out_features // 4, 1),
            BLD(
                out_features // 4,
                out_features // 4,
                stride=stride,
                dilation=dilation,
                groups=out_features // 4,
            ),
            BLD(out_features // 4,
                out_features,
                1,
                activate=False,
                se_block=se_block))
        self.downsample = EmptyLayer()
        if stride > 1 or in_features != out_features:
            self.downsample = BLD(in_features,
                                  out_features,
                                  3,
                                  stride,
                                  activate=False)

    def forward(self, x):
        return self.downsample(x) + self.block(x)
