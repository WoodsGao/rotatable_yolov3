import torch.nn as nn


class EmptyLayer(nn.Module):
    def forward(self, x):
        return x
