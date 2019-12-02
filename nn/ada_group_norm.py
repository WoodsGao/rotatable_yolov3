import torch.nn as nn
from . import EmptyLayer


class AdaGroupNorm(nn.Module):
    def __init__(self, channels):
        super(AdaGroupNorm, self).__init__()
        if channels % 4 != 0:
            self.norm = EmptyLayer()
        else:
            c = channels
            p = 1
            while c % 2 == 0 and c > 0:
                c /= 2
                p += 1
            p //= 2
            groups = 2**p
            self.norm = nn.GroupNorm(groups, channels)

    def forward(self, x):
        return self.norm(x)
