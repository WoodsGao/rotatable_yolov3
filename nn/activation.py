import torch
import torch.nn as nn
import torch.nn.functional as F


class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], 1)


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
