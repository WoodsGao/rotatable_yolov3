import os
import os.path as osp
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from imgaug import augmenters as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from pytorch_modules.utils import IMG_EXT
from pytorch_modules.datasets import LMDBCacheDataset

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
            rotate=(-15, 15),
            shear=(-0.1,
                   0.1)),  # rotate by -45 to 45 degrees (affects heatmaps)
        ia.ElasticTransformation(
            alpha=(0, 10),
            sigma=(0, 10)),  # apply water effect (affects heatmaps)
        ia.PiecewiseAffine(scale=(0, 0.03), nb_rows=(2, 6), nb_cols=(2, 6)),
        ia.GaussianBlur((0, 3)),
        ia.Fliplr(0.1),
        ia.Flipud(0.1),
        ia.LinearContrast((0.5, 1)),
        ia.AdditiveGaussianNoise(loc=(0, 10), scale=(0, 10))
    ],
    random_state=True)


class DetDataset(LMDBCacheDataset):
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

    def get_item(self, idx):
        img = cv2.imread(self.data[idx][0])
        h, w, c = img.shape
        if len(self.data[idx][1]):
            bboxes = np.float32(self.data[idx][1])
        else:
            bboxes = np.zeros([0, 5])

        classes = bboxes[:, 0]
        zeros = np.zeros_like(classes)
        bboxes = BoundingBoxesOnImage([
            BoundingBox(x1=int(b[0] * w),
                        y1=int(b[1] * h),
                        x2=int(b[2] * w),
                        y2=int(b[3] * h)) for b in bboxes[:, 1:]
        ],
                                      shape=img.shape)
        resize = self.resize.to_deterministic()
        img = resize.augment_image(img)
        bboxes = resize.augment_bounding_boxes(bboxes)

        # augment
        if self.augments is not None:
            augments = self.augments.to_deterministic()
            img = augments.augment_image(img)
            bboxes = augments.augment_bounding_boxes(bboxes)
            # print(bboxes)
            # image_after = bboxes.draw_on_image(img, thickness=2, color=[0, 0, 255])
            # cv2.imshow('a', image_after)
            # cv2.waitKey(0)

        bboxes = bboxes.to_xyxy_array()
        bboxes[:, 0] /= img.shape[1]
        bboxes[:, 2] /= img.shape[1]
        bboxes[:, 1] /= img.shape[0]
        bboxes[:, 3] /= img.shape[0]
        bboxes = bboxes.clip(0, 1)

        x = (bboxes[:, 0] + bboxes[:, 2]) / 2.
        y = (bboxes[:, 1] + bboxes[:, 3]) / 2.
        w = (bboxes[:, 2] - bboxes[:, 0])
        h = (bboxes[:, 3] - bboxes[:, 1])

        bboxes = np.stack([zeros, classes, x, y, w, h], 1)

        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return torch.ByteTensor(img), torch.FloatTensor(bboxes)

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
        imgs /= 255.
        
        if self.multi_scale:
            h = imgs.size(2)
            w = imgs.size(3)
            scale = random.uniform(0.7, 1.5)
            h = int(h * scale / 16) * 16
            w = int(w * scale / 16) * 16
            imgs = F.interpolate(imgs, (h, w))
        return (imgs, labels)
