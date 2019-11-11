from . import HSV, PerspectiveProject, Blur, Pepper
from random import random, uniform, randint, choice
import numpy as np


def augments_parser(cfg):
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
    matrix = np.eye(3)
    if 'rotate' in cfg:
        if random() < cfg['rotate']:
            angle = uniform(-1, 1) * np.pi
            matrix = np.dot(
                matrix,
                np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]))
    if 'translate' in cfg:
        if random() < cfg['translate']:
            translation = [
                cfg['size'] * 0.1 * uniform(-1, 1),
                cfg['size'] * 0.1 * uniform(-1, 1)
            ]
            matrix = np.dot(
                matrix,
                np.array([
                    [1, 0, translation[0]],
                    [0, 1, translation[1]],
                    [0, 0, 1],
                ]))
    if 'shear' in cfg:
        if random() < cfg['shear']:
            angle = uniform(-1, 1) * np.pi * 0.1
            matrix = np.dot(
                matrix,
                np.array([
                    [1, -np.sin(angle), 0],
                    [0, np.cos(angle), 0],
                    [0, 0, 1],
                ]))
    if 'scale' in cfg:
        if random() < cfg['scale']:
            factor = [uniform(0.8, 1.2), uniform(0.8, 1.2)]
            matrix = np.dot(
                matrix,
                np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]]))
    if 'flip' in cfg:
        if random() < cfg['flip']:
            factor = [choice([-1, 1]), choice([-1])]
            matrix = np.dot(
                matrix,
                np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]]))
    augments_list.append(PerspectiveProject(matrix))
    return augments_list
