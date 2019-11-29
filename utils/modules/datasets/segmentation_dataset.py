import cv2
import numpy as np
import os
import torch
from . import BasicDataset
from ..augments import augments_parser
from ..utils import IMG_EXT
import torch.nn.functional as F


def voc_colormap(N=256):
    def bitget(val, idx):
        return ((val & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << 7 - j)
            g |= (bitget(c, 1) << 7 - j)
            b |= (bitget(c, 2) << 7 - j)
            c >>= 3
        # print([r, g, b])
        cmap[i, :] = [b, g, r]
    return cmap


VOC_COLORMAP = voc_colormap(32)


class SegmentationDataset(BasicDataset):
    def build_data(self):
        data_dir = os.path.dirname(self.path)
        with open(os.path.join(data_dir, 'classes.names'), 'r') as f:
            self.classes = f.readlines()
        image_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')
        with open(self.path, 'r') as f:
            names = [n for n in f.read().split('\n') if n]
        names = list(set(names))
        self.data = [
            [
                os.path.join(image_dir, name),
                os.path.join(label_dir,
                             os.path.splitext(name)[0] + '.png')
            ] for name in names
            if os.path.splitext(name)[1] in IMG_EXT
        ]

    def get_item(self, idx):
        img = cv2.imread(self.data[idx][0])
        img = np.float32(img)
        seg_color = cv2.imread(self.data[idx][1])
        seg = np.zeros(
            [seg_color.shape[0], seg_color.shape[1],
             len(VOC_COLORMAP)])
        for ci, c in enumerate(VOC_COLORMAP):
            seg[(seg_color == c).all(2), ci] = 1
        for aug in augments_parser(self.augments, img.shape, self.img_size):
            img, _, seg = aug(img, seg=seg)
        img = img[:, :, ::-1]
        img = np.clip(img, 0, 255)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = np.uint8(img)
        seg[seg.sum(2) == 0, 0] = 1
        # seg = np.ascontiguousarray(seg.transpose(2, 0, 1))
        seg = seg.argmax(2)
        # for ci, c in enumerate(self.classes):
        #     seg[seg_args == ci, 1 if ci > 0 else 0] = 1
        return torch.ByteTensor(img), torch.ByteTensor(seg)

    @staticmethod
    def post_fetch_fn(batch):
        imgs, segs = batch
        imgs = imgs.float()
        imgs /= 255.
        segs = F.one_hot(segs.long(), 32).permute(0, 3, 1, 2).float()
        return (imgs, segs)
