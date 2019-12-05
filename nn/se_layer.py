import torch.nn as nn
from . import Swish


class SELayer(nn.Module):
    def __init__(self, filters):
        super(SELayer, self).__init__()
        div = 24 if filters >= 96 else 4
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Sequential(
            nn.Conv2d(filters, filters // div, 1),
            nn.ReLU(True),
            nn.Conv2d(filters // div, filters, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gap = self.gap(x)
        weight = self.weight(gap)
        return x * weight
