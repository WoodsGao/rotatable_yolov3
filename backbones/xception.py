import torch.nn as nn
from ..nn import DBL, Swish, XBlock, SeparableConv2d


class Xception(nn.Module):
    def __init__(self, output_stride=32):
        super(Xception, self).__init__()
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
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
        )
        self.block2 = nn.Sequential(
            XBlock(64, 128, reps=2, stride=2, start_with_relu=False),
            Swish(),
        )
        self.block3 = XBlock(128,
                             256,
                             reps=2,
                             stride=2,
                             start_with_relu=False,
                             grow_first=True)

        # Middle flow
        middle_flow = [
            XBlock(  #entry_block3
                256,
                728,
                reps=2,
                stride=entry_block3_stride,
                start_with_relu=True,
                grow_first=True,
                is_last=True)
        ]
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
            Swish(),
            SeparableConv2d(1024, 1536, 3, 1,
                            dilation=exit_block_dilations[1]),
            nn.BatchNorm2d(1536),
            Swish(),
            SeparableConv2d(1536,
                            1536,
                            3,
                            stride=1,
                            dilation=exit_block_dilations[1]),
            nn.BatchNorm2d(1536),
            Swish(),
            SeparableConv2d(1536, 2048, 3, 1,
                            dilation=exit_block_dilations[1]),
            nn.BatchNorm2d(2048),
            Swish(),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
