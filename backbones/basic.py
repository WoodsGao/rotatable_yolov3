import math
import torch
import torch.nn as nn
from ..nn import CNS, SeparableCNS


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.block1 = CNS(3, 32, 7, stride=2)
        self.block2 = SeparableCNS(32, 64, stride=2)
        self.block3 = SeparableCNS(64, 128, stride=2)
        self.block4 = SeparableCNS(128, 256, stride=2)
        self.block5 = SeparableCNS(256, 512, stride=2)
        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
