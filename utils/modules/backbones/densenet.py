import torch.nn as nn
from . import BasicModel
from ..nn import Swish, DenseBlock, CNS


class DenseNet(BasicModel):
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
            nn.Conv2d(3, 32, 7, 1, 3, bias=False),
            DenseBlock(32, 64, stride=2),
        )
        self.block2 = nn.Sequential(
            DenseBlock(64, 128, stride=2),
            DenseBlock(128, 128),
        )
        self.block3 = nn.Sequential(
            DenseBlock(128, 256, stride=2),
            DenseBlock(256, 256),
            DenseBlock(256, 256),
            DenseBlock(256, 256),
            DenseBlock(256, 256),
        )
        self.block4 = nn.Sequential(
            DenseBlock(256, 512, stride=block4_stride),
            DenseBlock(512, 512),
            DenseBlock(512, 512),
            DenseBlock(512, 512),
            DenseBlock(512, 512),
            DenseBlock(512, 512),
            DenseBlock(512, 512),
            DenseBlock(512, 512),
            DenseBlock(512, 512),
        )
        self.block5 = nn.Sequential(
            DenseBlock(512, 1024, stride=block5_stride),
            DenseBlock(1024, 1024),
            DenseBlock(1024, 1024),
        )
