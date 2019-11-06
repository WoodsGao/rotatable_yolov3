import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat([relu(x), relu(-x)], 1)


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class SELayer(nn.Module):
    def __init__(self, filters):
        super(SELayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Sequential(
            nn.Conv2d(filters, filters // 16 if filters >= 32 else 8, 1),
            nn.LeakyReLU(inplace=True),
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
    def __init__(self, in_channels, out_channels):
        super(AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class Aspp(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(Aspp, self).__init__()
        blocks = []
        blocks.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
            ))
        for rate in atrous_rates:
            blocks.append(DBL(in_channels, out_channels, dilation=rate))
        blocks.append(AsppPooling(in_channels, out_channels))
        self.blocks = nn.ModuleList(blocks)
        self.project = nn.Sequential(
            DBL(out_channels * len(blocks), out_channels, 1), nn.Dropout(0.5))

    def forward(self, x):
        outputs = []
        for block in self.blocks:
            outputs.append(block(x))
        x = torch.cat(outputs, 1)
        x = self.project(x)
        return x


class BLD(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 groups=1,
                 dilation=1,
                 activate=Swish(),
                 se_block=False):
        super(BLD, self).__init__()
        if activate is None:
            activate = EmptyLayer()
        if se_block:
            se_block = SELayer(in_channels)
        else:
            se_block = EmptyLayer()
        self.bld = nn.Sequential(
            nn.BatchNorm2d(in_channels), se_block, activate,
            nn.Conv2d(in_channels *
                      2 if isinstance(activate, CReLU) else in_channels,
                      out_channels,
                      ksize,
                      stride=stride,
                      padding=(ksize - 1) // 2 - 1 + dilation,
                      groups=groups,
                      dilation=dilation,
                      bias=False))

    def forward(self, x):
        return self.bld(x)


class DBL(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 groups=1,
                 dilation=1,
                 activate=nn.LeakyReLU(inplace=True),
                 se_block=False):
        super(DBL, self).__init__()
        if activate is None:
            activate = EmptyLayer()
        if se_block:
            se_block = SELayer(out_channels)
        else:
            se_block = EmptyLayer()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels *
                      2 if isinstance(activate, CReLU) else in_channels,
                      out_channels,
                      ksize,
                      stride=stride,
                      padding=(ksize - 1) // 2 - 1 + dilation,
                      groups=groups,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            activate,
            se_block,
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, filters, dilation=1, se_block=False):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            BLD(filters, filters // 4, 1),
            BLD(
                filters // 4,
                filters // 4,
                dilation=dilation,
                groups=filters // 4,
            ),
            BLD(filters // 4, filters, 1, se_block=se_block),
        )

    def forward(self, x):
        return x + self.block(x)


class ResBackbone(nn.Module):
    def __init__(self, output_stride=32):
        super(ResBackbone, self).__init__()
        assert output_stride in [8, 16, 32]
        if output_stride == 32:
            block4_stride = 2
            block5_stride = 2
        elif output_stride == 16:
            block4_stride = 2
            block5_stride = 1
        elif output_stride == 8:
            block4_stride = 1
            block5_stride = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            ResBlock(32),
        )
        self.block1 = nn.Sequential(
            BLD(32, 64, stride=2),
            ResBlock(64),
        )
        self.block2 = nn.Sequential(
            BLD(64, 128, stride=2),
            ResBlock(128),
            ResBlock(128),
        )
        self.block3 = nn.Sequential(
            BLD(128, 256, stride=2),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
        )
        self.block4 = nn.Sequential(
            BLD(256, 512, stride=block4_stride),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
        )
        self.block5 = nn.Sequential(
            BLD(512, 1024, stride=block5_stride),
            ResBlock(1024),
            ResBlock(1024),
            ResBlock(1024),
            ResBlock(1024),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


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
                rep.append(nn.LeakyReLU())
            rep.append(
                SeparableConv2d(in_channels, out_channels, 3, 1, dilation))
            rep.append(nn.BatchNorm2d(out_channels))
            filters = out_channels
        for i in range(reps - 1):
            if grow_first or start_with_relu:
                rep.append(nn.LeakyReLU())
            rep.append(SeparableConv2d(
                filters,
                filters,
                3,
                1,
                dilation,
            ))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(nn.LeakyReLU(inplace=True))
            rep.append(
                SeparableConv2d(
                    in_channels,
                    out_channels,
                    3,
                    1,
                    dilation,
                ))
        if stride != 1:
            rep.append(nn.LeakyReLU(inplace=True))
            rep.append(SeparableConv2d(
                out_channels,
                out_channels,
                3,
                stride,
            ))
            rep.append(nn.BatchNorm2d(out_channels))
        elif is_last:
            rep.append(nn.LeakyReLU(inplace=True))
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


class XceptionBackbone(nn.Module):
    def __init__(self, output_stride=32):
        super(XceptionBackbone, self).__init__()
        assert output_stride in [8, 16, 32]
        if output_stride == 32:
            entry_block3_stride = 2
            exit_block20_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 1)
        elif output_stride == 16:
            entry_block3_stride = 2
            exit_block20_stride = 1
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            exit_block20_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        # Entry flow
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            XBlock(64, 128, reps=2, stride=2, start_with_relu=False),
            nn.LeakyReLU(inplace=True),
        )
        self.block2 = XBlock(128,
                             256,
                             reps=2,
                             stride=2,
                             start_with_relu=False,
                             grow_first=True)
        self.block3 = XBlock(256,
                             728,
                             reps=2,
                             stride=entry_block3_stride,
                             start_with_relu=True,
                             grow_first=True,
                             is_last=True)
        # Middle flow
        middle_flow = list()
        for i in range(16):
            middle_flow.append(
                XBlock(728,
                       728,
                       reps=3,
                       stride=1,
                       dilation=middle_block_dilation,
                       start_with_relu=True,
                       grow_first=True))
        self.block4 = nn.Sequential(*middle_flow)

        # Exit flow
        self.block5 = nn.Sequential(
            XBlock(728,
                   1024,
                   reps=2,
                   stride=exit_block20_stride,
                   dilation=exit_block_dilations[0],
                   start_with_relu=True,
                   grow_first=False,
                   is_last=True),
            nn.LeakyReLU(inplace=True),
            SeparableConv2d(1024, 1536, 3, 1,
                            dilation=exit_block_dilations[1]),
            nn.BatchNorm2d(1536),
            nn.LeakyReLU(inplace=True),
            SeparableConv2d(1536,
                            1536,
                            3,
                            stride=1,
                            dilation=exit_block_dilations[1]),
            nn.BatchNorm2d(1536),
            nn.LeakyReLU(inplace=True),
            SeparableConv2d(1536, 2048, 3, 1,
                            dilation=exit_block_dilations[1]),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


if __name__ == "__main__":
    a = torch.ones([2, 3, 224, 224])
    b = ResBackbone(8)(a)
    print(b.shape)
    b.mean().backward()