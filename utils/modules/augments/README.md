# image_augments

## Introduction

An image augmentation tool based on OpenCV3 and Numpy, which can be used for data preprocessing, and can be used with dataloader as data generator.

## Features

 - Achieved data augmentation of the image/bounding box/segmentation data with one instance
 - Combine a series of processes
 - Use opencv and numpy
 - High performance

## Installation

### As a subtree

    git remote add image_augments https://github.com/woodsgao/image_augments 
    git subtree add --prefix=<subtree_path> image_augments master
    cd <subtree_path>
    pip install -r requirements.txt

### As a repository

    git clone https://github.com/woodsgao/image_augments
    cd image_augments
    pip install -r requirements.txt

## Usage

**image_augments** provides a series of data augmentation classes, and a configuration parser. You can write a dict type configuration and pass it to the configuration parser for parsing. It will return a list of data augmentation instances. Iterate through the list and use the image/bounding box/segmentation as parameters to get the augmented data.

    import numpy as np
    from image_augments import augments_parser

    augments = {
        'hsv': 0.1,
        'blur': 0.1,
        'pepper': 0.1,
        'shear': 0.1,
        'translate': 0.1,
        'rotate': 0.1,
        'flip': 0.1,
        'scale': 0.1,
        'noise': 0.1,
    }
    # image (H, W, channels)
    img = np.zeros([180, 224, 3], dtype=np.uint8)
    # segmentation (H, W, classes)
    seg = np.zeros([180, 224, 22], dtype=np.uint8)
    seg[:, :, 0] = 1
    # bounding box (N, (x1 y1 x2 y2 x3 y3 x4 y4))
    det = [0, 0, 0, 180, 224, 180, 224, 0]

    for aug in augments_parser(augments, img.shape, 224)):
        img, det, seg = aug(img, det, seg)
    