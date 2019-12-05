import torch.nn as nn
from . import BasicModel
from ..nn import ResBlock, CNS


class ResNet(BasicModel):
    def __init__(self):
        super(ResNet, self).__init__()
        self.block1 = nn.Sequential(
            CNS(3, 32, 7),
            ResBlock(32, 64, stride=2, reps=1),
        )
        self.block2 = ResBlock(64, 128, stride=2, reps=2)
        self.block3 = ResBlock(128, 256, stride=2, reps=8)
        self.block4 = ResBlock(256, 512, stride=2, reps=8)
        self.block5 = ResBlock(512, 1024, stride=2, reps=4)
