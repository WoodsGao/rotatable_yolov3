import torch
import torch.nn as nn
import torch.nn.functional as F
from . import SeparableCNS, CNS


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
        first_conv = []
        for idx, channels in enumerate(channels_list):
            first_conv.append(CNS(channels, out_channels, 1))
        self.first_conv = nn.ModuleList(first_conv)
        
        fpn_list = []
        for i in range(reps):
            fpn_stage = []
            for idx, channels in enumerate(channels_list):
                fpn_stage.append(SeparableCNS(out_channels, out_channels))
            fpn_stage = nn.ModuleList(fpn_stage)
            fpn_list.append(fpn_stage)
        self.fpn_list = nn.ModuleList(fpn_list)

    def forward(self, features):
        features = [conv(f) for conv, f in zip(self.first_conv, features)]
        for fpn in self.fpn_list:
            new_features = []
            for idx, (feature,
                      stage) in enumerate(zip(features[::-1], fpn)):
                if idx > 0:
                    last_feature = new_features[-1]
                    last_feature = F.interpolate(last_feature,
                                                 scale_factor=2,
                                                 mode='bilinear',
                                                 align_corners=True)
                    feature += last_feature
                feature = stage(feature)
                new_features.append(feature)
            features = new_features[::-1]
        return features
