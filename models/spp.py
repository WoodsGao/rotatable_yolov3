import torch
import torch.nn as nn


class SPP(nn.Module):
    def __init__(self, sizes=[5, 9, 13]):
        super(SPP, self).__init__()
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(size, 1, padding=(size - 1) // 2) for size in sizes])

    def forward(self, x):
        features = [x]
        for pool in self.pools:
            features.append(pool(x))
        features = torch.cat(features, 1)
        return features
