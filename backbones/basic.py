import math
import torch
import torch.nn as nn
from ..nn import CNS


class BasicModel(nn.Module):
    def __init__(self, output_stride=32):
        super(BasicModel, self).__init__()
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
        self.init()
        self.weight_standard()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def weight_standard(self):
        for m in self.modules():
            if isinstance(m, CNS):
                for cm in m.modules():
                    if isinstance(cm, nn.Conv2d):
                        weight = cm.weight.data
                        weight_mean = weight.mean(dim=1, keepdim=True)
                        weight_mean = weight_mean.mean(dim=2, keepdim=True)
                        weight_mean = weight_mean.mean(dim=3, keepdim=True)
                        weight = weight - weight_mean

                        var = torch.var(weight.view(weight.size(0), -1),
                                        dim=1,
                                        unbiased=False).clamp(min=1e-3)
                        std = torch.sqrt(var).view(-1, 1, 1, 1)
                        cm.weight.data = weight / std.expand_as(weight)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
