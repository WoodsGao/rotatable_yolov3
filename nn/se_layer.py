import torch.nn as nn
from . import Swish


class SELayer(nn.Module):
    def __init__(self, filters):
        super(SELayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Sequential(
            nn.Conv2d(filters, filters // 16 if filters >= 32 else 8, 1),
            Swish(),
            nn.Conv2d(filters // 16 if filters >= 32 else 8, filters, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gap = self.gap(x)
        weight = self.weight(gap)
        return x * weight
