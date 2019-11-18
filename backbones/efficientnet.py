import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nn import MbConv, CNS


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


class EfficientNetB2(nn.Module):
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