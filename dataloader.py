import cv2
import numpy as np
import os
import random
from threading import Thread
import time
from augments import *
from det_utils import *
from queue import Queue


class ClsDataloader(object):

    def __init__(self, path, img_size=224, batch_size=8, augments=[], balance=True, multi_scale=False):
        self.path = path
        self.img_size = img_size
        self.batch_size = batch_size
        self.augments = augments
        self.multi_scale = multi_scale
        self.classes = os.listdir(path)
        self.classes.sort()
        self.data_list = list()

        if balance:
            weights = [len(os.listdir(os.path.join(path, c)))
                       for c in self.classes]
            max_weight = max(weights)
        for i_c, c in enumerate(self.classes):
            names = os.listdir(os.path.join(path, c))
            if balance:
                names *= (max_weight // len(names))+1
                names = names[:max_weight]
            for name in names:
                self.data_list.append([os.path.join(path, c, name), i_c])
        self.iter_times = len(self.data_list) // self.batch_size + 1
        self.max_len = 50
        self.queue = []
        self.scale = img_size
        self.batch_list = []
        t = Thread(target=self.run)
        t.setDaemon(True)
        t.start()

    def __iter__(self):
        return self

    def worker(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.scale, self.scale))
        for aug in self.augments:
            img, _, __ = aug(img)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(
            img, dtype=np.float32)  # uint8 to float32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img -= np.mean(img)
        img /= np.std(img)
        return img

    def run(self):
        while True:
            while len(self.batch_list) > self.max_len:
                time.sleep(0.1)
            if len(self.queue) == 0:
                random.shuffle(self.data_list)
                self.queue = self.data_list

            if self.multi_scale:
                self.scale = random.randint(
                    min(self.img_size//40, 1), self.img_size//20) * 32
                # print(self.scale)
            its = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]
            imgs = [self.worker(it[0]) for it in its]
            self.batch_list.append(
                [np.float32(imgs), np.int64([it[1] for it in its])])

    def next(self):
        while len(self.batch_list) == 0:
            time.sleep(0.1)
        batch = self.batch_list[0]
        self.batch_list = self.batch_list[1:]

        return batch[0], batch[1]


class DetDataloader(object):

    def __init__(self, path, img_size=224, batch_size=8, augments=[], multi_scale=False):
        self.path = path
        self.img_size = img_size
        self.batch_size = batch_size
        self.augments = augments
        self.multi_scale = multi_scale

        with open(path, 'r') as f:
            self.data_list = f.read().split('\n')
        self.data_list = [d for d in self.data_list if d]

        self.iter_times = len(self.data_list) // self.batch_size + 1
        self.queue = list()
        self.scale = img_size
        self.batch_list = Queue(maxsize=20)
        t = Thread(target=self.run)
        t.setDaemon(True)
        t.start()

    def __iter__(self):
        return self

    def worker(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.scale, self.scale))
        with open(path.replace('images', 'labels').replace('png', 'txt').replace('jpg', 'txt'), 'r') as f:
            det = f.read().split('\n')
            det = [[float(di) for di in d.split(' ')] for d in det if d]

        # det: n*(c x y w h)
        if len(det) > 0:
            det = np.float32(det)
            det[:, 1:] *= self.scale
            # points n*(x1 y1 x2 y2 x3 y3 x4 y4)
            classes = det[:, 0:1]
            det = xywh2points(det[:, 1:])
        else:
            det = None

        for aug in self.augments:
            img, det, __ = aug(img, det=det)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(
            img, dtype=np.float32)  # uint8 to float32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img -= np.mean(img)
        img /= np.std(img)
        if det is not None:
            # n*(x, y, w, h)
            # det /= self.scale
            # det[det>1] = 1
            det[det>self.scale] = self.scale
            det[det<0] = 0
            det = points2ltrb(det)
            det = ltrb2xywh(det)
            det = np.concatenate([classes, det], 1)
        return img, det

    def run(self):
        while True:
            if len(self.queue) == 0:
                random.shuffle(self.data_list)
                self.queue = self.data_list

            if self.multi_scale:
                self.scale = random.randint(
                    min(self.img_size//40, 1), self.img_size//20) * 32
                # print(self.scale)

            its = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]
            results = [self.worker(it) for it in its]
            imgs = np.float32([r[0] for r in results])
            targets = [[np.insert(l, 0, ir) for l in r[1]]
                       for ir, r in enumerate(results) if r[1] is not None]
            if len(targets) > 0:
                targets = np.concatenate(targets, 0)
            else:
                targets = np.zeros([0, 6])
            targets = np.float32(targets)
            self.batch_list.put([imgs, targets])

    def next(self):
        while len(self.batch_list.queue) == 0:
            time.sleep(0.1)

        return self.batch_list.get()


class SegDataloader(object):

    def __init__(self, path, num_classes, img_size=224,  batch_size=8, augments=[], multi_scale=True):
        self.path = path
        self.num_classes = num_classes
        self.img_size = img_size
        self.batch_size = batch_size
        self.augments = augments
        self.multi_scale = multi_scale
        self.data_list = list()
        names = os.listdir(os.path.join(path, 'images'))
        for name in names:
            self.data_list.append(
                [os.path.join(path, 'images', name), os.path.join(path, 'seg', name)])
        self.iter_times = len(self.data_list) // self.batch_size + 1
        self.queue = []
        self.scale = 1

    def __iter__(self):
        return self

    def worker(self, paths):
        # print(paths)
        seg = cv2.imread(paths[1])
        img = cv2.imread(paths[0])
        seg = cv2.resize(seg, (int(self.img_size*self.scale//32)
                               * 32, int(self.img_size*self.scale//32)*32))
        img = cv2.resize(img, (int(self.img_size*self.scale//32)
                               * 32, int(self.img_size*self.scale//32)*32))
        for aug in self.augments:
            img = aug(img)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
        print(img[127, 186], img[186, 127])
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(
            img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        seg = seg[:, :, 0:1]
        seg[seg > 125] = 255
        seg[seg < 125] = 0
        seg[seg > 0] = 1
        # print(seg.shape)
        # seg /= 255
        # canvas = np.zeros([int(self.img_size*self.scale//32)*32, int(self.img_size*self.scale//32)*32, self.num_classes])
        # canvas[seg>125] = [1, 0]
        # canvas[seg<=125] = [0, 1]
        return [img, seg.transpose(2, 0, 1)]

    def next(self):
        if len(self.queue) == 0:
            random.shuffle(self.data_list)
            self.queue = self.data_list

        # if self.multi_scale:
        #     self.scale = random.random()+1.5

        its = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]

        with ThreadPoolExecutor() as pool:
            results = pool.map(self.worker, its)
        results = list(results)
        return np.float32([r[0] for r in results]), np.int64([r[1] for r in results])


if __name__ == "__main__":
    augments_list = [
        PerspectiveProject(0.4, 0.7),
        HSV_H(0.1, 0.7),
        HSV_S(0.1, 0.7),
        HSV_V(0.1, 0.7),
        Rotate(1, 0.7),
        Blur(0.1, 0.3),
        Noise(0.05, 0.3),
    ]
    d = DetDataloader('data/mark/train.txt',
                      augments=augments_list)
    for i in range(d.iter_times):
        print(d.next())
