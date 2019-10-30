from .cv_utils import dataloader
import os
import numpy as np
import cv2


class Dataloader(dataloader.Dataloader):
    def build_data_list(self):
        self.classes = os.listdir(self.path)
        self.classes.sort()
        for ci, c in enumerate(self.classes):
            names = os.listdir(os.path.join(self.path, c))
            for name in names:
                target = np.zeros(len(self.classes))
                target[ci] = 1
                self.data_list.append(
                    [os.path.join(self.path, c, name), target])

    def worker(self, message, scale):
        img = cv2.imread(message[0])
        img = cv2.resize(img, (scale, scale))
        for aug in self.augments:
            img, _, __ = aug(img)
        return img, message[1]


def show_batch(save_path, messages, scale, classes, augments_list=[]):
    imgs = []
    for message in messages:
        img = cv2.imread(message[0])
        img = cv2.resize(img, (scale, scale))
        for aug in augments_list:
            img, _, __ = aug(img)
        imgs.append(img)
    save_img = np.concatenate(imgs, 1)
    save_img = np.clip(save_img, 0, 255)
    save_img = np.uint8(save_img)
    cv2.imwrite(save_path, save_img)
