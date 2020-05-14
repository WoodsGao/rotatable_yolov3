import json
import math
import os
import os.path as osp
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from imgaug import augmenters as ia
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

from pytorch_modules.utils import IMG_EXT

TRAIN_AUGS = ia.SomeOf(
    [0, 3],
    [
        ia.WithColorspace(
            to_colorspace='HSV',
            from_colorspace='RGB',
            children=ia.Sequential([
                ia.WithChannels(
                    0,
                    ia.SomeOf([0, None],
                              [ia.Add((-10, 10)),
                               ia.Multiply((0.95, 1.05))],
                              random_state=True)),
                ia.WithChannels(
                    1,
                    ia.SomeOf([0, None],
                              [ia.Add((-50, 50)),
                               ia.Multiply((0.8, 1.2))],
                              random_state=True)),
                ia.WithChannels(
                    2,
                    ia.SomeOf([0, None],
                              [ia.Add((-50, 50)),
                               ia.Multiply((0.8, 1.2))],
                              random_state=True)),
            ])),
        ia.Dropout([0.015, 0.1]),  # drop 5% or 20% of all pixels
        ia.Sharpen((0.0, 1.0)),  # sharpen the image
        ia.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-90, 90),
            shear=(-0.1,
                   0.1)),  # rotate by -45 to 45 degrees (affects heatmaps)
        # ia.ElasticTransformation(
        #     alpha=(0, 10),
        #     sigma=(0, 10)),  # apply water effect (affects heatmaps)
        # ia.PiecewiseAffine(scale=(0, 0.03), nb_rows=(2, 6), nb_cols=(2, 6)),
        ia.GaussianBlur((0, 3)),
        ia.Fliplr(0.1),
        ia.Flipud(0.1),
        # ia.LinearContrast((0.5, 1)),
        ia.AdditiveGaussianNoise(loc=(0, 10), scale=(0, 10))
    ],
    random_state=True)


class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, img_size, augments, multi_scale, rect, with_label,
                 mosaic):
        super(BasicDataset, self).__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        assert len(img_size) == 2
        self.img_size = img_size
        self.rect = rect
        self.multi_scale = multi_scale
        self.augments = augments
        self.with_label = with_label
        self.mosaic = mosaic
        self.data = []

    def get_data(self, idx):
        return None, None

    def __getitem__(self, idx):
        img, polygons = self.get_item(idx)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img, polygons = torch.ByteTensor(img), torch.FloatTensor(polygons)
        return img, polygons

    def get_item(self, idx, mosaic=True):
        img, polygons = self.get_data(idx)
        img = img[..., ::-1]
        h, w, c = img.shape

        if self.rect:
            scale = min(self.img_size[0] / w, self.img_size[1] / h)
            resize = ia.Sequential([
                ia.Resize({
                    'width': int(w * scale),
                    'height': int(h * scale)
                }),
                ia.PadToFixedSize(*self.img_size,
                                  pad_cval=[123.675, 116.28, 103.53],
                                  position='center')
            ])
        else:
            resize = ia.Resize({
                'width': self.img_size[0],
                'height': self.img_size[1]
            })

        img = resize.augment_image(img)
        polygons = resize.augment_polygons(polygons)
        # augment
        if self.augments is not None:
            augments = self.augments.to_deterministic()
            img = augments.augment_image(img)
            polygons = augments.augment_polygons(polygons)

        labels = []
        for polygon in polygons.polygons:
            p = polygon.exterior.reshape(-1, 2)
            p[:, 0] /= img.shape[1]
            p[:, 1] /= img.shape[0]
            p = p.clip(0, 1)
            c = polygon.label
            labels.append([0, c] + p.reshape(-1).tolist())
        if len(labels):
            labels = np.float32(labels)
        else:
            labels = np.zeros([0, 10], dtype=np.float32)
        if self.mosaic and mosaic:
            mosaic_list = [(img, labels)]
            for _ in range(3):
                next_idx = random.randint(0, self.__len__() - 1)
                mosaic_list.append(self.get_item(next_idx, False))
            cut_x_ratio = random.uniform(0.2, 0.8)
            cut_y_ratio = random.uniform(0.2, 0.8)
            cut_x = int(cut_x_ratio * self.img_size[0])
            cut_y = int(cut_y_ratio * self.img_size[1])
            random.shuffle(mosaic_list)
            img0 = mosaic_list[0][0][:cut_y, :cut_x]
            img1 = mosaic_list[1][0][:cut_y, cut_x:]
            img0 = np.concatenate([img0, img1], 1)
            img2 = mosaic_list[2][0][cut_y:, :cut_x]
            img3 = mosaic_list[3][0][cut_y:, cut_x:]
            img2 = np.concatenate([img2, img3], 1)
            img = np.concatenate([img0, img2], 0)
            labels0 = mosaic_list[0][1]
            labels0[:, 2] = labels0[:, 2].clip(0, cut_x_ratio)
            labels0[:, 3] = labels0[:, 3].clip(0, cut_y_ratio)
            labels1 = mosaic_list[1][1]
            labels1[:, 2] = labels1[:, 2].clip(cut_x_ratio, 1)
            labels1[:, 3] = labels1[:, 3].clip(0, cut_y_ratio)
            labels2 = mosaic_list[2][1]
            labels2[:, 2] = labels2[:, 2].clip(0, cut_x_ratio)
            labels2[:, 3] = labels2[:, 3].clip(cut_y_ratio, 1)
            labels3 = mosaic_list[3][1]
            labels3[:, 2] = labels3[:, 2].clip(cut_x_ratio, 1)
            labels3[:, 3] = labels3[:, 3].clip(cut_y_ratio, 1)
            labels = np.concatenate([labels0, labels1, labels2, labels3], 0)
        # filter small items
        labels = labels[labels[:, 2::2].max(1) - labels[:, 2::2].min(1) > 3e-3]
        labels = labels[labels[:, 3::2].max(1) - labels[:, 3::2].min(1) > 3e-3]
        if self.mosaic and not mosaic:
            return img, labels
        x, y, w, h, theta = [], [], [], [], []
        for l in labels:
            polygon = l[2:]
            xy, wh, t = cv2.minAreaRect(polygon.reshape(4, 1, 2))
            # t /= 180 / math.pi
            # t += 90
            # t /= 90.
            x.append(xy[0])
            y.append(xy[1])
            if wh[0] < wh[1]:
                w_ = wh[1]
                h_ = wh[0]
                t += 90
            else:
                w_ = wh[0]
                h_ = wh[1]
            w.append(w_)
            h.append(h_)
            theta.append(t)
        # print(theta)
        # labels = np.stack([labels[:, 0], labels[:, 1], x, y, w, h], 1)
        labels = np.stack([labels[:, 0], labels[:, 1], x, y, w, h, theta], 1)
        # print(labels[:, 4].mean() * 416, labels[:, 5].mean() * 416)
        labels[:, -1] /= 180. / np.pi
        return img, labels

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        imgs, dets = list(zip(*batch))  # transposed

        for i, l in enumerate(dets):
            l[:, 0] = i  # add target image index for build_targets()
        dets = torch.cat(dets, 0)
        imgs = torch.stack(imgs, 0)
        return imgs, dets

    def post_fetch_fn(self, batch):
        imgs, labels = batch
        imgs = imgs.float()
        imgs -= torch.FloatTensor([123.675, 116.28,
                                   103.53]).reshape(1, 3, 1, 1).to(imgs.device)
        imgs /= torch.FloatTensor([58.395, 57.12,
                                   57.375]).reshape(1, 3, 1, 1).to(imgs.device)

        if self.multi_scale:
            h = imgs.size(2)
            w = imgs.size(3)
            scale = random.uniform(0.7, 1.5)
            h = int(h * scale / 32) * 32
            w = int(w * scale / 32) * 32
            imgs = F.interpolate(imgs, (h, w))
        return (imgs, labels)


class CocoDataset(BasicDataset):
    def __init__(self,
                 path,
                 img_size=224,
                 augments=TRAIN_AUGS,
                 multi_scale=False,
                 rect=False,
                 with_label=False,
                 mosaic=False):
        super(CocoDataset, self).__init__(img_size=img_size,
                                          augments=augments,
                                          multi_scale=multi_scale,
                                          rect=rect,
                                          with_label=with_label,
                                          mosaic=mosaic)
        with open(path, 'r') as f:
            self.coco = json.loads(f.read())
        self.img_root = osp.dirname(path)
        self.augments = augments
        self.classes = []
        self.build_data()
        self.data.sort()

    def build_data(self):
        img_ids = []
        img_paths = []
        img_anns = []
        self.classes = [c['name'] for c in self.coco['categories']]
        for img_info in self.coco['images']:
            img_ids.append(img_info['id'])
            img_paths.append(osp.join(self.img_root, img_info['file_name']))
            img_anns.append([])
        for ann in self.coco['annotations']:
            idx = ann['image_id']
            idx = img_ids.index(idx)
            img_anns[idx].append(ann)
        self.data = list(zip(img_paths, img_anns))
        if self.with_label:
            self.data = [d for d in self.data if len(d[1]) > 0]

    def get_data(self, idx):
        img = cv2.imread(self.data[idx][0])
        anns = self.data[idx][1]
        polygons = []
        for ann in anns:
            polygons.append(
                Polygon(
                    np.float32(ann['segmentation']).reshape(-1, 2),
                    ann['category_id']))
        polygons = PolygonsOnImage(polygons, img.shape)
        return img, polygons
