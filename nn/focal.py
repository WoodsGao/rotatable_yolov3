import torch
import torch.nn as nn


class FocalBCELoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 weight=None,
                 reduction='mean',
                 eps=1e-5):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        if weight is not None:
            self.weight = weight.unsqueeze(0).unsqueeze(2)
        else:
            self.weight = 1

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.eps, 1 - self.eps)
        a = self.alpha
        g = self.gamma
        loss = - a * torch.pow((1 - y_pred), g) * y_true * torch.log(y_pred) - \
            (1 - a) * torch.pow(y_pred, g) * (1 - y_true) * torch.log(1 - y_pred)
        loss *= self.weight
        mloss = loss.sum(1)
        if self.reduction == 'mean':
            return mloss.mean()
        if self.reduction == 'sum':
            return mloss.sum()
        else:
            return mloss
