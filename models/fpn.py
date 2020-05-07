import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_modules.nn.utils import ConvNormAct, SeparableConvNormAct


class FPN(nn.Module):
    def __init__(self, channels_list, out_channels=128, reps=3):
        """[summary]
        
        Arguments:
            channels_list {list} -- channels of feature maps, from high levels to low levels
        
        Keyword Arguments:
            out_channels {int} -- out channels  (default: {128})
            reps {int} -- repeat times (default: {3})
        """
        super(FPN, self).__init__()
        self.fpn_list = nn.ModuleList([])
        last_channels = 0
        for i in range(len(channels_list)):
            in_channels = channels_list[i] + last_channels
            fpn = [SeparableConvNormAct(in_channels, out_channels)]
            fpn += [SeparableConvNormAct(out_channels, out_channels)] * (reps - 1)
            fpn = nn.Sequential(*fpn)
            self.fpn_list.append(fpn)
            last_channels = out_channels

    def forward(self, features):
        new_features = []
        for i in range(len(self.fpn_list)):
            feature = features[i]
            if len(new_features):
                last_feature = new_features[-1]
                last_feature = F.interpolate(last_feature,
                                             scale_factor=2,
                                             mode='bilinear',
                                             align_corners=False)
                feature = torch.cat([last_feature, feature], 1)
            feature = self.fpn_list[i](feature)
            new_features.append(feature)
        return new_features
