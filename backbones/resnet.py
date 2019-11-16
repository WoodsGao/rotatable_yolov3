import torch.nn as nn
from ..nn import ResBlock


class ResNet(nn.Module):
    def __init__(self, output_stride=32):
        super(ResNet, self).__init__()
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
            nn.Conv2d(3, 64, 7, 1, 3, bias=False),
            ResBlock(64, 128, stride=2),
        )
        self.block2 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
        )
        self.block3 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
        )
        self.block4 = nn.Sequential(
            ResBlock(512, 1024, stride=block4_stride),
            ResBlock(1024, 1024),
            ResBlock(1024, 1024),
            ResBlock(1024, 1024),
            ResBlock(1024, 1024),
            ResBlock(1024, 1024),
            ResBlock(1024, 1024),
        )
        self.block5 = nn.Sequential(
            ResBlock(1024, 2048, stride=block5_stride),
            ResBlock(2048, 2048),
            ResBlock(2048, 2048),
            ResBlock(2048, 2048),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
