import torch.nn as nn
from . import SELayer, Swish, SeparableConv2d, EmptyLayer


class XBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 reps=2,
                 stride=1,
                 dilation=1,
                 start_with_relu=True,
                 grow_first=True,
                 is_last=False,
                 se_block=False):
        super(XBlock, self).__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = EmptyLayer()
        rep = list()
        filters = in_channels
        if grow_first:
            if start_with_relu:
                rep.append(Swish())
            rep.append(
                SeparableConv2d(in_channels, out_channels, 3, 1, dilation))
            rep.append(nn.BatchNorm2d(out_channels))
            filters = out_channels
        for i in range(reps - 1):
            if grow_first or start_with_relu:
                rep.append(Swish())
            rep.append(SeparableConv2d(
                filters,
                filters,
                3,
                1,
                dilation,
            ))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(Swish())
            rep.append(
                SeparableConv2d(
                    in_channels,
                    out_channels,
                    3,
                    1,
                    dilation,
                ))
        if stride != 1:
            rep.append(Swish())
            rep.append(SeparableConv2d(
                out_channels,
                out_channels,
                3,
                stride,
            ))
            rep.append(nn.BatchNorm2d(out_channels))
        elif is_last:
            rep.append(Swish())
            rep.append(
                SeparableConv2d(
                    out_channels,
                    out_channels,
                    3,
                    1,
                    dilation,
                ))
            rep.append(nn.BatchNorm2d(out_channels))
        if se_block:
            rep.append(SELayer(out_channels))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        x = self.skip(x) + self.rep(x)
        return x
