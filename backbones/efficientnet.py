import torch.nn as nn
import math
from . import BasicModel
from ..nn import MbBlock, CNS

params_list = [
    (1, 1, 0.2),
    (1, 1.1, 0.2),
    (1.1, 1.2, 0.3),
    (1.2, 1.4, 0.3),
    (1.4, 1.8, 0.4),
    (1.6, 2.2, 0.4),
    (1.8, 2.6, 0.5),
    (2, 3.1, 0.5),
]


class EfficientNet(BasicModel):
    def __init__(self, model_id=2):
        super(EfficientNet, self).__init__()
        params = params_list[model_id]
        width = [32, 16, 24, 40, 80, 112, 192, 320]
        width = [math.ceil(w * params[0] / 8) * 8 for w in width]
        depth = [1, 2, 2, 3, 3, 4, 1]
        depth = [math.ceil(d * params[1]) for d in depth]
        self.block1 = nn.Sequential(
            CNS(3, width[0], stride=2),
            MbBlock(width[0],
                    width[1],
                    expand_ratio=1,
                    reps=depth[0],
                    drop_rate=params[2]),
        )
        self.block2 = nn.Sequential(
            MbBlock(width[1],
                    width[2],
                    stride=2,
                    reps=depth[1],
                    drop_rate=params[2]), )
        self.block3 = nn.Sequential(
            MbBlock(width[2],
                    width[3],
                    5,
                    stride=2,
                    reps=depth[2],
                    drop_rate=params[2]), )
        self.block4 = nn.Sequential(
            MbBlock(width[3],
                    width[4],
                    3,
                    stride=2,
                    reps=depth[3],
                    drop_rate=params[2]),
            MbBlock(width[4], width[5], 5, reps=depth[4], drop_rate=params[2]),
        )
        self.block5 = nn.Sequential(
            MbBlock(width[5],
                    width[6],
                    5,
                    stride=2,
                    reps=depth[5],
                    drop_rate=params[2]),
            MbBlock(width[6], width[7], 3, reps=depth[6], drop_rate=params[2]),
        )
        self.width = width
        self.depth = depth
