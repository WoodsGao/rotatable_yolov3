from . import HSV, PerspectiveProject, Blur, Pepper, Noise
from random import random, uniform, randint, choice
import numpy as np
import cv2


def augments_parser(cfg, src_size, dst_size):
    augments_list = []
    if 'hsv' in cfg:
        if random() < cfg['hsv']:
            augments_list.append(
                HSV([
                    0.01 * uniform(-1, 1), 0.7 * uniform(-1, 1),
                    0.4 * uniform(-1, 1)
                ]))
    if 'blur' in cfg:
        if random() < cfg['blur']:
            augments_list.append(Blur(randint(1, 3) * 2 + 1))
    if 'pepper' in cfg:
        if random() < cfg['pepper']:
            augments_list.append(Pepper(randint(1, 3) * 2 + 1))
    if 'noise' in cfg:
        if random() < cfg['noise']:
            augments_list.append(Noise(randint(10, 100)))

    # project
    factor = dst_size / max(src_size)
    translation = [(dst_size - factor * src_size[1]) / 2,
                   (dst_size - factor * src_size[0]) / 2]
    matrix = np.float32([
        [factor, 0, translation[0]],
        [0, factor, translation[1]],
        [0, 0, 1],
    ])
    if 'rotate' in cfg:
        if random() < cfg['rotate']:
            angle = uniform(-1, 1) * np.pi
            matrix = np.dot(
                np.array([
                    [
                        np.cos(angle),
                        -np.sin(angle),
                        (1 - np.cos(angle) + np.sin(angle)) * 0.5 * dst_size,
                    ],
                    [
                        np.sin(angle),
                        np.cos(angle),
                        (1 - np.cos(angle) - np.sin(angle)) * 0.5 * dst_size,
                    ],
                    [0, 0, 1],
                ]), matrix)
    if 'shear' in cfg:
        if random() < cfg['shear']:
            angle = uniform(-1, 1) * np.pi * 0.1
            matrix = np.dot(
                np.array([
                    [1, -np.sin(angle),
                     np.sin(angle) * 0.5 * dst_size],
                    [0, np.cos(angle), (1 - np.cos(angle)) * 0.5 * dst_size],
                    [0, 0, 1],
                ]), matrix)
    if 'scale' in cfg:
        if random() < cfg['scale']:
            factor = [uniform(0.8, 1.1), uniform(0.8, 1.1)]
            matrix = np.dot(
                np.array([[factor[0], 0, (1 - factor[0]) * 0.5 * dst_size],
                          [0, factor[1], (1 - factor[1]) * 0.5 * dst_size],
                          [0, 0, 1]]), matrix)
    if 'flip' in cfg:
        if random() < cfg['flip']:
            factor = [choice([-1, 1]), choice([-1])]
            matrix = np.dot(
                np.array([[factor[0], 0, (1 - factor[0]) * 0.5 * dst_size],
                          [0, factor[1], (1 - factor[1]) * 0.5 * dst_size],
                          [0, 0, 1]]), matrix)
    if 'translate' in cfg:
        if random() < cfg['translate']:
            translation = [
                dst_size * 0.1 * uniform(-1, 1),
                dst_size * 0.1 * uniform(-1, 1)
            ]
            matrix = np.dot(
                np.array([
                    [1, 0, translation[0]],
                    [0, 1, translation[1]],
                    [0, 0, 1],
                ]), matrix)
    augments_list.append(PerspectiveProject(matrix, (dst_size, dst_size)))
    return augments_list
