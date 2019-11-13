import cv2
import numpy as np
import os
import torch
from . import BasicDataset
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
        classes = [b[0] for b in bboxes]
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
        bboxes = []
        for c, d in zip(classes, det):
            xs = d[0::2]
            ys = d[1::2]
            xmax = xs.max()
            xmin = xs.min()
            ymax = ys.max()
            ymin = ys.min()
            if ymax - ymin < 3. / h or xmax - xmin < 3. / w:
                continue
            x = (xmax + xmin) / 2
            y = (ymax + ymin) / 2
            w = x - xmin
            h = y - ymin
            bboxes.append([0, c, x, y, w, h])
        bboxes_tensor = torch.zeros((len(bboxes), 6))
        if len(bboxes):
            bboxes_tensor = torch.FloatTensor(bboxes)
        img = img[:, :, ::-1]
        img /= 255.
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return torch.FloatTensor(img), bboxes_tensor, self.data[idx][0], (h, w)

    @staticmethod
    def collate_fn(batch):
        img, dets, path, hw = list(zip(*batch))  # transposed
        for i, l in enumerate(dets):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(dets, 0), path, hw
