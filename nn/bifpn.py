import torch
import torch.nn as nn
import torch.nn.functional as F
from . import SeparableCNS


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
        assert reps > 0
        self.td_weights = torch.ones([reps, len(channels_list), 2],
                                     requires_grad=True)
        self.out_weights = torch.ones([reps, len(channels_list), 3],
                                      requires_grad=True)
        conv_td_list = []
        conv_out_list = []
        for i in range(reps):
            conv_td = []
            conv_out = []
            for idx, channels in enumerate(channels_list):
                if idx == len(channels_list) - 1:
                    td_in = channels
                else:
                    td_in = channels + out_channels

                if idx == 0:
                    out_in = out_channels
                else:
                    out_in = channels + 2 * out_channels
                conv_td.append(SeparableCNS(td_in, out_channels))
                conv_out.append(SeparableCNS(out_in, out_channels))
                channels_list[idx] = out_channels
            conv_td = nn.ModuleList(conv_td)
            conv_out = nn.ModuleList(conv_out)
            conv_td_list.append(conv_td)
            conv_out_list.append(conv_out)
        self.conv_td_list = nn.ModuleList(conv_td_list)
        self.conv_out_list = nn.ModuleList(conv_out_list)
        self.eps = eps

    def forward(self, features):
        self.td_weights = self.td_weights.relu()
        self.td_weights = self.td_weights / (
            self.td_weights.sum(2, keepdim=True) + self.eps)
        self.out_weights = self.out_weights.relu()
        self.out_weights = self.out_weights / (
            self.out_weights.sum(2, keepdim=True) + self.eps)
        for li, (conv_td, conv_out) in enumerate(
                zip(self.conv_td_list, self.conv_out_list)):
            ftd_list = []
            for idx, (f, conv) in enumerate(zip(features[::-1],
                                                conv_td[::-1])):
                ftd = f
                if idx > 0:
                    ftd = ftd * self.td_weights[li, idx, 0]
                    high = ftd_list[-1]
                    high = F.interpolate(high,
                                         scale_factor=2,
                                         mode='bilinear',
                                         align_corners=True)
                    high *= self.td_weights[li, idx, 1]
                    ftd = torch.cat([ftd, high], 1)
                ftd = conv(ftd)
                ftd_list.append(ftd)
            ftd_list = ftd_list[::-1]
            fout_list = []
            for idx, (f, ftd,
                      conv) in enumerate(zip(features, ftd_list, conv_out)):
                fout = ftd
                if idx > 0:
                    f = f * self.out_weights[li, idx, 0]
                    ftd = ftd * self.out_weights[li, idx, 1]
                    low = fout_list[-1]
                    low = F.interpolate(low,
                                        scale_factor=0.5,
                                        mode='bilinear',
                                        align_corners=True)
                    low = low * self.out_weights[li, idx, 2]
                    ftd = torch.cat([f, ftd, low], 1)
                ftd = conv(ftd)
                fout_list.append(fout)
            features = fout_list

        return features
