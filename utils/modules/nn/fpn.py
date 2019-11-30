import torch
import torch.nn as nn
import torch.nn.functional as F
from . import SeparableCNS


class FPN(nn.Module):
    def __init__(self, channels_list, out_channels=128, reps=3):
        """[summary]
        
        Arguments:
            channels_list {list} -- channels of feature maps, from low levels to high levels
        
        Keyword Arguments:
            out_channels {int} -- out channels  (default: {128})
            reps {int} -- repeat times (default: {3})
        """
        super(FPN, self).__init__()
        assert reps > 0
        fpn_stage = []
        for idx, channels in enumerate(channels_list[::-1]):
            if idx == 0:
                fpn_stage.append(SeparableCNS(channels, out_channels))
            else:
                fpn_stage.append(
                    SeparableCNS(channels + out_channels, out_channels))
        fpn_stage = nn.ModuleList(fpn_stage[::-1])
        fpn_list = [fpn_stage]
        for i in range(reps - 1):
            fpn_stage = []
            for idx, channels in enumerate(channels_list[::-1]):
                if idx == 0:
                    fpn_stage.append(
                        SeparableCNS(out_channels, out_channels))
                else:
                    fpn_stage.append(
                        SeparableCNS(2 * out_channels, out_channels))
            fpn_stage = nn.ModuleList(fpn_stage[::-1])
            fpn_list.append(fpn_stage)
        self.fpn_list = nn.ModuleList(fpn_list)

    def forward(self, features):
        for fpn in self.fpn_list:
            new_features = []
            for idx, (feature, stage) in enumerate(zip(features[::-1], fpn[::-1])):
                if idx > 0:
                    last_feature = new_features[-1]
                    last_feature = F.interpolate(last_feature,
                                                 scale_factor=2,
                                                 mode='bilinear',
                                                 align_corners=True)
                    feature = torch.cat([last_feature, feature], 1)
                feature = stage(feature)
                new_features.append(feature)
            features = new_features[::-1]
        return features
