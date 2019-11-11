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
    
    return augments_list
