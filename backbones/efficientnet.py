import torch.nn as nn
from . import BasicModel
from ..nn import MbBlock, CNS


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
            MbBlock(64, 32, expand_ratio=1, reps=2, drop_rate=0.5),
        )
        self.block2 = nn.Sequential(
            MbBlock(32, 48, stride=2, reps=7, drop_rate=0.5), )
        self.block3 = nn.Sequential(
            MbBlock(48, 80, 5, stride=2, reps=7, drop_rate=0.5), )
        self.block4 = nn.Sequential(
            MbBlock(80, 160, 5, stride=block4_stride, reps=10, drop_rate=0.5),
            MbBlock(160, 224, 5, reps=10, drop_rate=0.5),
        )
        self.block5 = nn.Sequential(
            MbBlock(224, 384, 5, stride=block5_stride, reps=13, drop_rate=0.5),
            MbBlock(384, 640, 3, reps=4, drop_rate=0.5),
        )


class EfficientNetB4(BasicModel):
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
            MbBlock(48, 24, expand_ratio=1, reps=2, drop_rate=0.4),
        )
        self.block2 = nn.Sequential(
            MbBlock(24, 32, stride=2, reps=4, drop_rate=0.4), )
        self.block3 = nn.Sequential(
            MbBlock(32, 56, 5, stride=2, reps=4, drop_rate=0.4), )
        self.block4 = nn.Sequential(
            MbBlock(56, 112, stride=block4_stride, reps=6, drop_rate=0.4),
            MbBlock(112, 160, 5, reps=6, drop_rate=0.4),
        )
        self.block5 = nn.Sequential(
            MbBlock(160, 272, 5, stride=block5_stride, reps=8, drop_rate=0.4),
            MbBlock(272, 448, 3, reps=2, drop_rate=0.4),
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
            MbBlock(32, 16, expand_ratio=1, reps=2, drop_rate=0.3),
        )
        self.block2 = nn.Sequential(
            MbBlock(16, 24, stride=2, reps=3, drop_rate=0.3), )
        self.block3 = nn.Sequential(
            MbBlock(24, 48, 5, stride=2, reps=3, drop_rate=0.3), )
        self.block4 = nn.Sequential(
            MbBlock(48, 88, stride=block4_stride, reps=4, drop_rate=0.3),
            MbBlock(88, 120, 5, reps=4, drop_rate=0.3),
        )
        self.block5 = nn.Sequential(
            MbBlock(120, 208, 5, stride=block5_stride, reps=4, drop_rate=0.3),
            MbBlock(208, 352, 3, reps=1, drop_rate=0.3),
        )