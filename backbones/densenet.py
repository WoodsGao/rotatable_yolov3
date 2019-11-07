import torch.nn as nn
from ..nn import BLD, Swish, DenseBlock, DBL


class DenseNet(nn.Module):
    def __init__(self, output_stride=32):
        super(DenseNet, self).__init__()
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

        self.block1 = nn.Sequential(
            DBL(3, 32, 7),
            DBL(32, 64, stride=2),
        )
        self.block2 = nn.Sequential(
            DBL(64, 128, stride=2),
            ResBlock(128),
            nn.BatchNorm2d(128),
            Swish(),
        )
        self.block3 = nn.Sequential(
            DBL(128, 256, stride=2),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            nn.BatchNorm2d(256),
            Swish(),
        )
        self.block4 = nn.Sequential(
            DBL(256, 512, stride=block4_stride),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            nn.BatchNorm2d(512),
            Swish(),
        )
        self.block5 = nn.Sequential(
            DBL(512, 1024, stride=block5_stride),
            ResBlock(1024),
            ResBlock(1024),
            nn.BatchNorm2d(1024),
            Swish(),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x