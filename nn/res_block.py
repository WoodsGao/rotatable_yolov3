import torch.nn as nn
from . import BLD


class ResBlock(nn.Module):
    def __init__(self, filters, dilation=1, se_block=False):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            BLD(filters, filters // 4, 1),
            BLD(
                filters // 4,
                filters // 4,
                dilation=dilation,
                groups=filters // 4,
            ),
            BLD(filters // 4, filters, 1, se_block=se_block),
        )

    def forward(self, x):
        return x + self.block(x)

