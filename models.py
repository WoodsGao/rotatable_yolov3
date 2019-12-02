import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.modules.nn import CNS, Swish, MbBlock, FPN, SeparableCNS, BiFPN, AdaGroupNorm
from utils.modules.backbones import BasicModel, EfficientNet
import math

# from utils.utils import *

ONNX_EXPORT = False


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints

        if ONNX_EXPORT:  # grids must be computed in __init__
            stride = [32, 16, 8][yolo_index]  # stride of this layer
            nx = int(img_size[1] / stride)  # number x grid points
            ny = int(img_size[0] / stride)  # number y grid points
            create_grids(self, img_size, (nx, ny))

    def forward(self, bbox_feature, cls_feature, img_size, var=None):
        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, ny, nx = bbox_feature.shape[0], bbox_feature.shape[-2], bbox_feature.shape[-1]
            if (self.nx, self.ny) != (nx, ny):
                create_grids(self, img_size, (nx, ny), bbox_feature.device, bbox_feature.dtype)

        bbox_feature = bbox_feature.view(bs, self.na, 4, self.ny, self.nx)
        cls_feature = cls_feature.view(bs, self.na, 1 + self.nc, self.ny, self.nx)
        p = torch.cat([bbox_feature, cls_feature], 2)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Constants CAN NOT BE BROADCAST, ensure correct shape!
            ngu = self.ng.repeat((1, self.na * self.nx * self.ny, 1))
            grid_xy = self.grid_xy.repeat((1, self.na, 1, 1, 1)).view(
                (1, -1, 2))
            anchor_wh = self.anchor_wh.repeat(
                (1, 1, self.nx, self.ny, 1)).view((1, -1, 2)) / ngu

            p = p.view(-1, 5 + self.nc)
            xy = torch.sigmoid(p[..., 0:2]) + grid_xy[0]  # x, y
            wh = torch.exp(p[..., 2:4]) * anchor_wh[0]  # width, height
            p_conf = torch.sigmoid(p[:, 4:5])  # Conf
            p_cls = F.softmax(p[:, 5:85], 1) * p_conf  # SSD-like conf
            return torch.cat((xy / ngu[0], wh, p_conf, p_cls), 1).t()

            # p = p.view(1, -1, 5 + self.nc)
            # xy = torch.sigmoid(p[..., 0:2]) + grid_xy  # x, y
            # wh = torch.exp(p[..., 2:4]) * anchor_wh  # width, height
            # p_conf = torch.sigmoid(p[..., 4:5])  # Conf
            # p_cls = p[..., 5:5 + self.nc]
            # # Broadcasting only supported on first dimension in CoreML. See onnx-coreml/_operators.py
            # # p_cls = F.softmax(p_cls, 2) * p_conf  # SSD-like conf
            # p_cls = torch.exp(p_cls).permute((2, 1, 0))
            # p_cls = p_cls / p_cls.sum(0).unsqueeze(0) * p_conf.permute((2, 1, 0))  # F.softmax() equivalent
            # p_cls = p_cls.permute(2, 1, 0)
            # return torch.cat((xy / ngu, wh, p_conf, p_cls), 2).squeeze().t()

        else:  # inference
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(
                io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., :4] *= self.stride

            torch.sigmoid_(io[..., 4:])

            if self.nc == 1:
                io[...,
                   5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return io.view(bs, -1, 5 + self.nc), p


class YOLOV3(BasicModel):
    # YOLOv3 object detection model

    def __init__(self, num_classes, img_size=512):
        super(YOLOV3, self).__init__()
        ratios = np.float32([1, 2, 3, 1 / 2, 1 / 3])
        ratios = np.sqrt(ratios)
        w_ratios = ratios
        h_ratios = 1 / ratios
        default_anchors = np.float32([w_ratios, h_ratios]).transpose(1, 0)
        model_id = 2
        self.backbone = EfficientNet(model_id)
        width = [416, 512, 608, 704]
        width = [int(w * (1.1**model_id) / 8) * 8 for w in width]
        self.backbone.width += width
        self.backbone.out_channels += [width[1], width[3]]
        depth = [5, 1, 6, 1]
        depth = [int(d * (1.2**model_id)) for d in depth]
        self.backbone.depth += depth
        self.backbone.block6 = nn.Sequential(
            MbBlock(self.backbone.width[7],
                    self.backbone.width[8],
                    5,
                    stride=2,
                    reps=self.backbone.depth[7],
                    drop_rate=self.backbone.drop_ratio),
            MbBlock(self.backbone.width[8],
                    self.backbone.width[9],
                    3,
                    reps=self.backbone.depth[8],
                    drop_rate=self.backbone.drop_ratio),
        )
        self.backbone.block7 = nn.Sequential(
            MbBlock(self.backbone.width[9],
                    self.backbone.width[10],
                    5,
                    stride=2,
                    reps=self.backbone.depth[9],
                    drop_rate=self.backbone.drop_ratio),
            MbBlock(self.backbone.width[10],
                    self.backbone.width[11],
                    3,
                    reps=self.backbone.depth[10],
                    drop_rate=self.backbone.drop_ratio),
        )
        width = int(8 * (1.35**model_id)) * 8
        self.fpn = BiFPN(self.backbone.out_channels[2:], width, 2 + model_id)
        bbox_conv = []
        cls_conv = []
        for i in range(3 + model_id // 3):
            bbox_conv.append(SeparableCNS(width, width))
            cls_conv.append(SeparableCNS(width, width))
        bbox_conv.append(nn.Conv2d(width, len(default_anchors) * 4, 1))
        cls_conv.append(
            nn.Conv2d(width,
                      len(default_anchors) * (1 + num_classes), 1))
        self.bbox_conv = nn.Sequential(*bbox_conv)
        self.cls_conv = nn.Sequential(*cls_conv)
        yolo_layers = []
        for i in range(3, 8):
            yolo_layers.append(
                YOLOLayer(
                    anchors=(2**i) * default_anchors,
                    nc=num_classes,
                    img_size=img_size,
                    yolo_index=i - 3,
                ))
        self.yolo_layers = nn.ModuleList(yolo_layers)
        self.init()
        self.num_classes = num_classes
        self.num_anchors = len(default_anchors)

    def forward(self, x):
        img_size = x.shape[-2:]
        output = []
        features = []
        x = self.backbone.block1(x)
        x = self.backbone.block2(x)
        x = self.backbone.block3(x)
        features.append(x)
        x = self.backbone.block4(x)
        features.append(x)
        x = self.backbone.block5(x)
        features.append(x)
        x = self.backbone.block6(x)
        features.append(x)
        x = self.backbone.block7(x)
        features.append(x)
        features = self.fpn(features)
        for fi, feature in enumerate(features):
            bbox_feature = self.bbox_conv(feature)
            cls_feature = self.cls_conv(feature)
            output.append(self.yolo_layers[fi](bbox_feature, cls_feature,
                                               img_size))
        if self.training:
            return output
        elif ONNX_EXPORT:
            output = torch.cat(
                output, 1)  # cat 3 layers 85 x (507, 2028, 8112) to 85 x 1064
            return output[5:5 + self.num_classes].t(), output[:4].t(
            )  # ONNX scores, boxes
        else:
            io, p = list(zip(*output))  # inference output, training output
            return torch.cat(io, 1), p


def create_grids(self,
                 img_size=416,
                 ng=(13, 13),
                 device='cpu',
                 type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view(
        (1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1,
                                          2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny


if __name__ == '__main__':
    model = YOLOV3(80)
    a = torch.rand([4, 3, 128, 128])
    b = model(a)
    print(b[0].shape)
    b[0].mean().backward()
