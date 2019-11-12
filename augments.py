import cv2
import random
import numpy as np


# bigger better
class PerspectiveProject:
    def __init__(self, matrix, shape):
        self.matrix = matrix
        self.shape = shape

    def __call__(self, img, det=None, seg=None):
        img = cv2.warpPerspective(img, self.matrix, self.shape)
        if det is not None:
            new_det = list()
            # detection n*(x1 y1 x2 y2 x3 y3 x4 y4)
            for i in range(4):
                point = np.concatenate(
                    [det[:, 2 * i:2 * i + 2],
                     np.ones([det.shape[0], 1])], 1)
                point = np.dot(self.matrix,
                               point.transpose(1, 0)).transpose(1, 0)
                point[:, :2] /= point[:, 2:]
                new_det.append(point[:, :2])
            det = np.concatenate(new_det, 1)
        if seg is not None:
            seg = cv2.warpPerspective(seg, self.matrix, self.shape)
        return img, det, seg


class HSV:
    def __init__(self, rate=[0.01, 0.7, 0.4]):
        self.rate = np.float32([[rate]])

    def __call__(self, img, det=None, seg=None):
        img = np.clip(img, 0, 255)
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = np.float32(img)
        img += img * self.rate
        img = np.clip(img, 0, 255)
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = np.float32(img)
        return img, det, seg


class Blur:
    def __init__(self, ksize=3):
        self.ksize = ksize

    def __call__(self, img, det=None, seg=None):
        img = cv2.blur(img, (self.ksize, self.ksize))
        return img, det, seg


class Pepper:
    def __init__(self, rate=0.05):
        self.rate = rate

    def __call__(self, img, det=None, seg=None):
        size = img.shape[0]
        amount = int(size * self.rate / 2 * random.random())

        x = np.random.randint(0, img.shape[0], amount)
        y = np.random.randint(0, img.shape[1], amount)
        img[x, y] = [0, 0, 0]

        x = np.random.randint(0, img.shape[0], amount)
        y = np.random.randint(0, img.shape[1], amount)
        img[x, y] = [255, 255, 255]
        return img, det, seg


class Noise:
    def __init__(self, rate=10):
        self.rate = rate

    def __call__(self, img, det=None, seg=None):
        noise = np.random.rand(*(img.shape))
        img += self.rate * noise
        img = np.clip(img, 0, 255)
        return img, det, seg
