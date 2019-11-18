import torch.nn as nn
from ..nn import CNS


class Mini(nn.Module):
    def __init__(self, output_stride=32):
        super(Mini, self).__init__()
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
        self.block1 = CNS(3, 64, stride=2)
        self.block2 = CNS(64, 128, stride=2)
        self.block3 = CNS(128, 256, stride=2)
        self.block4 = CNS(256, 512, stride=block4_stride)
        self.block5 = CNS(512, 1024, stride=block5_stride)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
