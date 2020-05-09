import json
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
        # ia.WithColorspace(
        #     to_colorspace='HSV',
        #     from_colorspace='RGB',
        #     children=ia.Sequential([
        #         ia.WithChannels(
        #             0,
        #             ia.SomeOf([0, None],
        #                       [ia.Add((-10, 10)),
        #                        ia.Multiply((0.95, 1.05))],
        #                       random_state=True)),
        #         ia.WithChannels(
        #             1,
        #             ia.SomeOf([0, None],
        #                       [ia.Add((-50, 50)),
        #                        ia.Multiply((0.8, 1.2))],
        #                       random_state=True)),
        #         ia.WithChannels(
        #             2,
        #             ia.SomeOf([0, None],
        #                       [ia.Add((-50, 50)),
        #                        ia.Multiply((0.8, 1.2))],
        #                       random_state=True)),
        #     ])),
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
        img, bboxes = self.get_item(idx)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img, bboxes = torch.ByteTensor(img), torch.FloatTensor(bboxes)
        bboxes[:, 4] -= bboxes[:, 2]
        bboxes[:, 5] -= bboxes[:, 3]
        bboxes[:, 2] += bboxes[:, 4] / 2.
        bboxes[:, 3] += bboxes[:, 5] / 2.
        return img, bboxes

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

        bboxes = []
        for polygon in polygons.polygons:
            p = polygon.exterior.reshape(-1, 2).transpose(1, 0)
            p[0] /= img.shape[1]
            p[1] /= img.shape[0]
            p = p.clip(0, 1)
            # # for line/point only
            if p.shape[1] <= 2:
                x1 = max(p[0].min() - 0.03, 0)
                x2 = min(p[0].max() + 0.03, 1)
                y1 = max(p[1].min() - 0.03, 0)
                y2 = min(p[1].max() + 0.03, 1)
            else:
                x1 = p[0].min()
                x2 = p[0].max()
                y1 = p[1].min()
                y2 = p[1].max()
            c = polygon.label
            bboxes.append([0, c, x1, y1, x2, y2])
        if len(bboxes):
            bboxes = np.float32(bboxes)
        else:
            bboxes = np.zeros([0, 6], dtype=np.float32)
        if self.mosaic and mosaic:
            mosaic_list = [(img, bboxes)]
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
            bboxes0 = mosaic_list[0][1]
            bboxes0[:, 2] = bboxes0[:, 2].clip(0, cut_x_ratio)
            bboxes0[:, 4] = bboxes0[:, 4].clip(0, cut_x_ratio)
            bboxes0[:, 3] = bboxes0[:, 3].clip(0, cut_y_ratio)
            bboxes0[:, 5] = bboxes0[:, 5].clip(0, cut_y_ratio)
            bboxes1 = mosaic_list[1][1]
            bboxes1[:, 2] = bboxes1[:, 2].clip(cut_x_ratio, 1)
            bboxes1[:, 4] = bboxes1[:, 4].clip(cut_x_ratio, 1)
            bboxes1[:, 3] = bboxes1[:, 3].clip(0, cut_y_ratio)
            bboxes1[:, 5] = bboxes1[:, 5].clip(0, cut_y_ratio)
            bboxes2 = mosaic_list[2][1]
            bboxes2[:, 2] = bboxes2[:, 2].clip(0, cut_x_ratio)
            bboxes2[:, 4] = bboxes2[:, 4].clip(0, cut_x_ratio)
            bboxes2[:, 3] = bboxes2[:, 3].clip(cut_y_ratio, 1)
            bboxes2[:, 5] = bboxes2[:, 5].clip(cut_y_ratio, 1)
            bboxes3 = mosaic_list[3][1]
            bboxes3[:, 2] = bboxes3[:, 2].clip(cut_x_ratio, 1)
            bboxes3[:, 4] = bboxes3[:, 4].clip(cut_x_ratio, 1)
            bboxes3[:, 3] = bboxes3[:, 3].clip(cut_y_ratio, 1)
            bboxes3[:, 5] = bboxes3[:, 5].clip(cut_y_ratio, 1)
            bboxes = np.concatenate([bboxes0, bboxes1, bboxes2, bboxes3], 0)
        # filter small items
        bboxes = bboxes[bboxes[:, 4] - bboxes[:, 2] > 3e-3]
        bboxes = bboxes[bboxes[:, 5] - bboxes[:, 3] > 3e-3]
        
        return img, bboxes

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


class YOLODataset(BasicDataset):
    def __init__(self,
                 path,
                 img_size=224,
                 augments=TRAIN_AUGS,
                 multi_scale=False,
                 rect=False,
                 with_label=True,
                 mosaic=False):
        super(YOLODataset, self).__init__(img_size=img_size,
                                          augments=augments,
                                          multi_scale=multi_scale,
                                          rect=rect,
                                          with_label=with_label,
                                          mosaic=mosaic)
        self.path = path
        self.classes = []
        self.build_data()
        self.data.sort()

    def build_data(self):
        data_dir = osp.dirname(self.path)
        with open(osp.join(data_dir, 'classes.names'), 'r') as f:
            self.classes = f.readlines()
        image_dir = osp.join(data_dir, 'images')
        label_dir = osp.join(data_dir, 'labels')
        with open(self.path, 'r') as f:
            names = [n for n in f.read().split('\n') if n]
        names = [name for name in names if osp.splitext(name)[1] in IMG_EXT]
        for name in names:
            bboxes = []
            label_name = osp.join(label_dir, osp.splitext(name)[0] + '.txt')
            if osp.exists(label_name):
                with open(label_name, 'r') as f:
                    lines = [[float(x) for x in l.split(' ') if x]
                             for l in f.readlines() if l]

                for l in lines:
                    if len(l) != 5:
                        continue
                    c = l[0]
                    x = l[1]
                    y = l[2]
                    w = l[3] / 2.
                    h = l[4] / 2.
                    xmin = x - w
                    xmax = x + w
                    ymin = y - h
                    ymax = y + h
                    if ymax > 1 or xmax > 1 or ymin > 1 or xmin > 1:
                        continue
                    if ymax < 0 or xmax < 0 or ymin < 0 or xmin < 0:
                        continue
                    if ymax <= ymin or xmax <= xmin:
                        continue
                    bboxes.append([c, xmin, ymin, xmax, ymax])
            self.data.append([osp.join(image_dir, name), bboxes])
            if self.with_label:
                self.data = [d for d in self.data if len(d[1]) > 0]

    def get_data(self, idx):
        img = cv2.imread(self.data[idx][0])
        polygons = []
        for c, xmin, ymin, xmax, ymax in self.data[idx][1]:
            polygons.append(
                Polygon(
                    np.float32(
                        [xmin, ymin, xmin, ymax, xmax, ymax, xmax,
                         ymin]).reshape(-1, 2), c))
        polygons = PolygonsOnImage(polygons, img.shape)


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
