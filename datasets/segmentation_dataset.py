import cv2
import numpy as np
import os
import torch
from . import BasicDataset


class SegmentationDataset(BasicDataset):
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
        self.data = [
            [
                os.path.join(image_dir, name),
                os.path.join(label_dir,
                             os.path.splitext(name)[0] + '.png')
            ] for name in names
            if os.path.splitext(name)[1] in ['jpg', 'jpeg', 'png', 'tiff']
        ]

    def get_item(self, idx):
        img = cv2.imread(self.data[idx][0])
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.float32(img)
        seg_color = cv2.imread(self.data[idx][1])
        seg = np.zeros(
            [seg_color.shape[0], seg_color.shape[1],
             len(self.classes)])
        for ci, c in enumerate(self.classes):
            seg[(seg_color == c[1]).all(2), ci] = 1
        seg = cv2.resize(seg, (self.img_size, self.img_size))
        for aug in self.augments:
            img, _, seg = aug(img, seg=seg)
        seg[seg.sum(2) == 0, 0] = 1
        seg_args = seg.argmax(0)
        # for ci, c in enumerate(self.classes):
        #     seg[seg_args == ci, 1 if ci > 0 else 0] = 1
        return torch.FloatTensor(img), torch.LongTensor(seg_args)
