from . import BasicDataset
from ..augments import augments_parser
from ..utils import IMG_EXT
import torch
import numpy as np
import os
import cv2
from torch.utils.data.dataloader import default_collate


class ClassificationDataset(BasicDataset):
    def build_data(self):
        self.data = []
        data_dir = os.path.dirname(self.path)
        class_names = [n for n in os.listdir(data_dir)]
        class_names = [
            cn for cn in class_names
            if os.path.isdir(os.path.join(data_dir, cn))
        ]
        class_names.sort()
        self.classes = class_names
        with open(self.path, 'r') as f:
            names = [n for n in f.read().split('\n') if n]
        self.data = [
            [
                os.path.join(data_dir, name),
                self.classes.index(os.path.basename(os.path.dirname(name)))
            ] for name in names
            if os.path.splitext(name)[1] in IMG_EXT
        ]

    def get_item(self, idx):
        img = cv2.imread(self.data[idx][0])
        img = np.float32(img)
        for aug in augments_parser(self.augments, img.shape, self.img_size):
            img, _, __ = aug(img)
        img = img[:, :, ::-1]
        img = np.clip(img, 0, 255)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = np.uint8(img)
        return torch.ByteTensor(img), self.data[idx][1]

    @staticmethod
    def post_fetch_fn(batch):
        imgs, labels = batch
        imgs = imgs.float()
        imgs /= 255.
        return (imgs, labels)
