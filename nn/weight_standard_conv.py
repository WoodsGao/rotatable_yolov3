import torch
import torch.nn as nn
import torch.nn.functional as F


# weight standard conv
# class WSConv2d(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super(WSConv2d, self).__init__(*args, **kwargs)
#         self.fused = False

#     def weight_standard(self):
#         weight = self.weight.data
#         weight_mean = weight.mean(dim=1, keepdim=True)
#         weight_mean = weight_mean.mean(dim=2, keepdim=True)
#         weight_mean = weight_mean.mean(dim=3, keepdim=True)
#         weight = weight - weight_mean

#         var = torch.var(weight.view(weight.size(0), -1), dim=1,
#                         unbiased=False).clamp(min=1e-12)
#         std = torch.sqrt(var).view(-1, 1, 1, 1)
#         weight = weight / std.expand_as(weight)
#         return weight

#     def fuse(self):
#         if not self.fused:
#             self.weight.data = self.weight_standard()
#             self.fused = True

#     def forward(self, x):
#         if self.training:
#             weight = self.weight_standard()
#             self.fused = False
#         elif self.fused:
#             weight = self.weight
#         else:
#             weight = self.weight_standard()
#         return F.conv2d(x, weight, self.bias, self.stride, self.padding,
#                         self.dilation, self.groups)

class WSConv2d(nn.Conv2d):
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True)
        weight_mean = weight_mean.mean(dim=2, keepdim=True)
        weight_mean = weight_mean.mean(dim=3, keepdim=True)
        weight = weight - weight_mean

        var = torch.var(weight.view(weight.size(0), -1), dim=1,
                        unbiased=False).clamp(min=1e-12)
        std = torch.sqrt(var).view(-1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)