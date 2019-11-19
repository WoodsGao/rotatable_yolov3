import torch.nn as nn
from . import BasicModel
from ..nn import MbConv, CNS


class EfficientNetB7(BasicModel):
    def __init__(self, output_stride=32):
        super(EfficientNetB7, self).__init__()
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
            CNS(3, 64, stride=2),
            MbConv(64, 32, expand_ratio=1),
            MbConv(32, 32, expand_ratio=2),
        )
        self.block2 = nn.Sequential(
            MbConv(32, 48, stride=2),
            MbConv(48, 48),
            MbConv(48, 48),
            MbConv(48, 48),
            MbConv(48, 48),
            MbConv(48, 48),
            MbConv(48, 48),
        )
        self.block3 = nn.Sequential(
            MbConv(48, 80, 5, stride=2),
            MbConv(80, 80, 5),
            MbConv(80, 80, 5),
            MbConv(80, 80, 5),
            MbConv(80, 80, 5),
            MbConv(80, 80, 5),
            MbConv(80, 80, 5),
        )
        self.block4 = nn.Sequential(
            MbConv(80, 160, 5, stride=block4_stride),
            MbConv(160, 160, 5),
            MbConv(160, 160, 5),
            MbConv(160, 160, 5),
            MbConv(160, 160, 5),
            MbConv(160, 160, 5),
            MbConv(160, 160, 5),
            MbConv(160, 160, 5),
            MbConv(160, 160, 5),
            MbConv(160, 160, 5),
            MbConv(160, 224, 5),
            MbConv(224, 224, 5),
            MbConv(224, 224, 5),
            MbConv(224, 224, 5),
            MbConv(224, 224, 5),
            MbConv(224, 224, 5),
            MbConv(224, 224, 5),
            MbConv(224, 224, 5),
            MbConv(224, 224, 5),
            MbConv(224, 224, 5),
        )
        self.block5 = nn.Sequential(
            MbConv(224, 384, 5, stride=block5_stride),
            MbConv(384, 384, 5),
            MbConv(384, 384, 5),
            MbConv(384, 384, 5),
            MbConv(384, 384, 5),
            MbConv(384, 384, 5),
            MbConv(384, 384, 5),
            MbConv(384, 384, 5),
            MbConv(384, 384, 5),
            MbConv(384, 384, 5),
            MbConv(384, 384, 5),
            MbConv(384, 384, 5),
            MbConv(384, 384, 5),
            MbConv(384, 640, 3),
            MbConv(640, 640, 3),
            MbConv(640, 640, 3),
            MbConv(640, 640, 3),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


class EfficientNetB4(nn.Module):
    def __init__(self, output_stride=32):
        super(EfficientNetB4, self).__init__()
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
            CNS(3, 48, stride=2),
            MbConv(48, 24, expand_ratio=1),
            MbConv(24, 24, expand_ratio=2),
        )
        self.block2 = nn.Sequential(
            MbConv(24, 32, stride=2),
            MbConv(32, 32),
            MbConv(32, 32),
            MbConv(32, 32),
        )
        self.block3 = nn.Sequential(
            MbConv(32, 56, 5, stride=2),
            MbConv(56, 56, 5),
            MbConv(56, 56, 5),
            MbConv(56, 56, 5),
        )
        self.block4 = nn.Sequential(
            MbConv(56, 112, stride=block4_stride),
            MbConv(112, 112),
            MbConv(112, 112),
            MbConv(112, 112),
            MbConv(112, 112),
            MbConv(112, 112),
            MbConv(112, 160, 5),
            MbConv(160, 160, 5),
            MbConv(160, 160, 5),
            MbConv(160, 160, 5),
            MbConv(160, 160, 5),
            MbConv(160, 160, 5),
        )
        self.block5 = nn.Sequential(
            MbConv(160, 272, 5, stride=block5_stride),
            MbConv(272, 272, 5),
            MbConv(272, 272, 5),
            MbConv(272, 272, 5),
            MbConv(272, 272, 5),
            MbConv(272, 272, 5),
            MbConv(272, 272, 5),
            MbConv(272, 272, 5),
            MbConv(272, 448, 3),
            MbConv(448, 448, 3),
        )


class EfficientNetB2(BasicModel):
    def __init__(self, output_stride=32):
        super(EfficientNetB2, self).__init__()
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
            CNS(3, 32, stride=2),
            MbConv(32, 16, expand_ratio=1),
            MbConv(16, 16, expand_ratio=2),
        )
        self.block2 = nn.Sequential(
            MbConv(16, 24, stride=2),
            MbConv(24, 24),
            MbConv(24, 24),
        )
        self.block3 = nn.Sequential(
            MbConv(24, 48, 5, stride=2),
            MbConv(48, 48, 5),
            MbConv(48, 48, 5),
        )
        self.block4 = nn.Sequential(
            MbConv(48, 88, stride=block4_stride),
            MbConv(88, 88),
            MbConv(88, 88),
            MbConv(88, 88),
            MbConv(88, 120, 5),
            MbConv(120, 120, 5),
            MbConv(120, 120, 5),
            MbConv(120, 120, 5),
        )
        self.block5 = nn.Sequential(
            MbConv(120, 208, 5, stride=block5_stride),
            MbConv(208, 208, 5),
            MbConv(208, 208, 5),
            MbConv(208, 208, 5),
            MbConv(208, 208, 5),
            MbConv(208, 352, 3),
            MbConv(352, 352, 3),
        )