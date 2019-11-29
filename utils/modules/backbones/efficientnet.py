import torch.nn as nn
from . import BasicModel
from ..nn import MbBlock, CNS


class EfficientNet(BasicModel):
    def __init__(self, model_id=2):
        super(EfficientNet, self).__init__()
        drop_ratio = 0.2 + 0.1 * (model_id // 2)
        width = [32, 16, 24, 40, 80, 112, 192, 320]
        width = [int(w * (1.1**model_id) / 8) * 8 for w in width]
        depth = [1, 2, 2, 3, 3, 4, 1]
        depth = [int(d * (1.2**model_id)) for d in depth]
        self.block1 = nn.Sequential(
            CNS(3, width[0], stride=2),
            MbBlock(width[0],
                    width[1],
                    expand_ratio=1,
                    reps=depth[0],
                    drop_rate=drop_ratio),
        )
        self.block2 = nn.Sequential(
            MbBlock(width[1],
                    width[2],
                    stride=2,
                    reps=depth[1],
                    drop_rate=drop_ratio), )
        self.block3 = nn.Sequential(
            MbBlock(width[2],
                    width[3],
                    5,
                    stride=2,
                    reps=depth[2],
                    drop_rate=drop_ratio), )
        self.block4 = nn.Sequential(
            MbBlock(width[3],
                    width[4],
                    3,
                    stride=2,
                    reps=depth[3],
                    drop_rate=drop_ratio),
            MbBlock(width[4], width[5], 5, reps=depth[4],
                    drop_rate=drop_ratio),
        )
        self.block5 = nn.Sequential(
            MbBlock(width[5],
                    width[6],
                    5,
                    stride=2,
                    reps=depth[5],
                    drop_rate=drop_ratio),
            MbBlock(width[6], width[7], 3, reps=depth[6],
                    drop_rate=drop_ratio),
        )
        self.width = width
        self.out_channels = [width[1], width[2], width[3], width[5], width[7]]
        self.depth = depth
        self.drop_ratio = drop_ratio
        self.img_size = int(7 * 1.15**model_id) * 32
