import torch.nn as nn
from . import NSC, EmptyLayer


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 se_block=False):
        super(ResBlock, self).__init__()
        if stride == 1 and in_channels == out_channels:
            self.downsample = EmptyLayer()
        else:
            self.downsample = NSC(in_channels, out_channels, stride=stride)
        self.block = nn.Sequential(
            NSC(in_channels, out_channels // 2, 1),
            NSC(
                out_channels // 2,
                out_channels // 2,
                stride=stride, 
                dilation=dilation,
                groups=32 if out_channels % 64 == 0 else 1,
            ),
<<<<<<< HEAD
            NSC(out_channels // 2, out_channels, 1),
=======
            BLD(out_channels // 2, out_channels, 1),
>>>>>>> bc8e130fe6ef2df7289bb59fec042ccad7bec48d
        )

    def forward(self, x):
        return self.downsample(x) + self.block(x)
