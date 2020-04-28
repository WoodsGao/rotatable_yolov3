import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_modules.backbones import resnet50
from pytorch_modules.nn import FPN, ConvNormAct
from pytorch_modules.utils import initialize_weights


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints

    def forward(self, p, img_size):
        bs, ny, nx = p.shape[0], p.shape[-2], p.shape[-1]
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.nc + 5, self.ny,
                   self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

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


class YOLOV3(nn.Module):
    # YOLOv3 object detection model

    def __init__(self,
                 num_classes,
                 img_size=(416, 416),
                 anchors=[
                     [[116, 90], [156, 198], [373, 326]],
                     [[30, 61], [62, 45], [59, 119]],
                     [[10, 13], [16, 30], [33, 23]],
                 ]):
        super(YOLOV3, self).__init__()
        self.stages = resnet50(pretrained=True).stages

        depth = 3
        width = 256
        planes_list = [512, 1024, 2048]
        self.fpn = FPN(planes_list, width, 3)
        self.head = nn.ModuleList([])
        self.yolo_layers = nn.ModuleList([])
        for i in range(3):
            head = []
            for j in range(depth - 1):
                head.append(ConvNormAct(width, width))
            head.append(
                nn.Conv2d(width,
                          len(anchors[i]) * (5 + num_classes), 1))
            self.head.append(nn.Sequential(*head))
            self.yolo_layers.append(
                YOLOLayer(
                    anchors=np.float32(anchors[i]),
                    nc=num_classes,
                    img_size=img_size,
                    yolo_index=i,
                ))
        initialize_weights(self.fpn)
        initialize_weights(self.head)

    def forward(self, x):
        img_size = x.shape[-2:]

        features = []
        x = self.stages[0](x)
        x = self.stages[1](x)
        x = self.stages[2](x)
        features.append(x)
        x = self.stages[3](x)
        features.append(x)
        x = self.stages[4](x)
        features.append(x)
        features = self.fpn(features)
        features.reverse()
        features = [
            head(feature) for feature, head in zip(features, self.head)
        ]
        if os.environ.get('CAFFE_EXPORT'):
            return None
        output = [
            yolo(feature, img_size)
            for feature, yolo in zip(features, self.yolo_layers)
        ]
        if self.training:
            return tuple(output)
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
    a = torch.rand([4, 3, 416, 416])
    b = model(a)
    print(b[0].shape)
    b[0].mean().backward()
