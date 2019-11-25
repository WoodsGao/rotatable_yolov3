import cv2
import numpy as np
import os
import torch
from . import BasicDataset, device
from ..augments import augments_parser


class DetectionDataset(BasicDataset):
    def build_data(self):
        data_dir = os.path.dirname(self.path)
        with open(os.path.join(data_dir, 'classes.names'), 'r') as f:
            lines = [l.split(',') for l in f.readlines()]
            lines = [[l[0], np.uint8(l[1:])] for l in lines if len(l) == 4]
        self.classes = lines
        image_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')
        with open(self.path, 'r') as f:
            names = [n for n in f.read().split('\n') if n]
        names = [
            name for name in names
            if os.path.splitext(name)[1] in ['.jpg', '.jpeg', '.png', '.tiff']
        ]
        for name in names:
            bboxes = []
            label_name = os.path.join(label_dir,
                                      os.path.splitext(name)[0] + '.txt')
            if os.path.exists(label_name):
                with open(label_name, 'r') as f:
                    lines = [[float(x) for x in l.split(' ') if x]
                             for l in f.readlines() if l]

                for l in lines:
                    if len(l) != 5:
                        continue
                    c = l[0]
                    x = l[1]
                    y = l[2]
                    w = l[3]
                    h = l[4]
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
                    bboxes.append(
                        [c, xmin, ymin, xmax, ymax, xmin, ymax, xmax, ymin])

            self.data.append([os.path.join(image_dir, name), bboxes])

    def get_item(self, idx):
        img = cv2.imread(self.data[idx][0])
        h, w, c = img.shape
        img = np.float32(img)
        bboxes = self.data[idx][1]
        classes = np.float32([b[0] for b in bboxes])
        det = np.zeros([len(bboxes), 8])
        for bi, b in enumerate(bboxes):
            det[bi] = b[1:]
        det[:, 0::2] *= w
        det[:, 1::2] *= h
        for aug in augments_parser(self.augments, img.shape, self.img_size):
            img, det, _ = aug(img, det=det)
        h, w, c = img.shape
        det[:, 0::2] /= w
        det[:, 1::2] /= h
        det = det.clip(0, 1)
        xs = det[:, ::2]
        ys = det[:, 1::2]
        xmax = xs.max(1)
        ymax = ys.max(1)
        xmin = xs.min(1)
        ymin = ys.min(1)
        x = (xmax + xmin) / 2
        y = (ymax + ymin) / 2
        w = x - xmin
        h = y - ymin
        zeros = np.zeros_like(classes)
        bboxes = np.stack([zeros, classes, x, y, w, h], 1)
        img = img[:, :, ::-1]
        img = np.clip(img, 0, 255)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = np.uint8(img)
        return torch.ByteTensor(img), torch.FloatTensor(bboxes), self.data[idx][0], (h, w)

    @staticmethod
    def collate_fn(batch):
        imgs, dets, path, hw = list(zip(*batch))  # transposed
        for i, l in enumerate(dets):
            l[:, 0] = i  # add target image index for build_targets()
        imgs = torch.stack(imgs, 0)
        imgs = imgs.float().to(device)
        imgs /= 255.
        return imgs, torch.cat(dets, 0), path, hw
