import torch
import torch.nn as nn
import torch.nn.functional as F
from . import SeparableCNS, CNS
from ..utils import device


class BiFPN(nn.Module):
    def __init__(self, channels_list, out_channels=128, reps=3, eps=1e-4):
        """[summary]
        
        Arguments:
            channels_list {list} -- channels of feature maps, from low levels to high levels
        
        Keyword Arguments:
            out_channels {int} -- out channels  (default: {128})
            reps {int} -- repeat times (default: {3})
            eps {int} -- a smalll number (default: {1e-4})
        """
        super(BiFPN, self).__init__()
        self.td_weights = nn.Parameter(
            torch.ones([reps, len(channels_list), 2]))
        self.out_weights = nn.Parameter(
            torch.ones([reps, len(channels_list), 3]))
        first_conv = []
        for idx, channels in enumerate(channels_list):
            first_conv.append(CNS(channels, out_channels, 1))
        self.first_conv = nn.ModuleList(first_conv)
        conv_td_list = []
        conv_out_list = []
        for i in range(reps):
            conv_td = []
            conv_out = []
            for idx, channels in enumerate(channels_list):
                conv_td.append(SeparableCNS(out_channels, out_channels))
                conv_out.append(SeparableCNS(out_channels, out_channels))
            conv_td = nn.ModuleList(conv_td)
            conv_out = nn.ModuleList(conv_out)
            conv_td_list.append(conv_td)
            conv_out_list.append(conv_out)
        self.conv_td_list = nn.ModuleList(conv_td_list)
        self.conv_out_list = nn.ModuleList(conv_out_list)
        self.eps = eps

    def forward(self, features):
        td_weights = self.td_weights.relu()
        td_weights = td_weights / (td_weights.sum(2, keepdim=True) + self.eps)
        out_weights = self.out_weights.relu()
        out_weights = out_weights / (out_weights.sum(2, keepdim=True) +
                                     self.eps)

        # first conv
        features = [
            conv(feature) for conv, feature in zip(self.first_conv, features)
        ]

        # BiFPN
        for li, (conv_td, conv_out) in enumerate(
                zip(self.conv_td_list, self.conv_out_list)):
            ftd_list = []
            for idx, (f, conv) in enumerate(zip(features[::-1],
                                                conv_td[::-1])):
                ftd = f
                if idx > 0:
                    ftd = ftd * td_weights[li, idx, 0]
                    high = ftd_list[-1]
                    high = F.interpolate(high,
                                         scale_factor=2,
                                         mode='bilinear',
                                         align_corners=True)
                    high *= td_weights[li, idx, 1]
                    ftd += high
                ftd = conv(ftd)
                ftd_list.append(ftd)
            ftd_list = ftd_list[::-1]
            fout_list = []
            for idx, (f, ftd,
                      conv) in enumerate(zip(features, ftd_list, conv_out)):
                fout = ftd
                if idx > 0:
                    f = f * out_weights[li, idx, 0]
                    ftd = ftd * out_weights[li, idx, 1]
                    low = fout_list[-1]
                    low = F.interpolate(low,
                                        scale_factor=0.5,
                                        mode='bilinear',
                                        align_corners=True)
                    low = low * out_weights[li, idx, 2]
                    ftd += f
                    ftd += low
                ftd = conv(ftd)
                fout_list.append(fout)
            features = fout_list
        return features
