# pytorch_modules

## Introduction

A neural network toolkit built on pytorch/opencv/numpy that includes neural network layers, modules, loss functions, optimizers, data loaders, data augmentation, etc.

## Features

 - Advanced neural network modules, such as ResNet, SENet, Xception, DenseNet, FocalLoss, AdaboundW
 - Ultra-efficient dataloader that allows you to take full advantage of GPU
 - High performance and multifunctional data augmentation(See [woodsgao/image_augments](https://github.com/woodsgao/image_augments))

## Installation

### As a subtree

    git remote add pytorch_modules https://github.com/woodsgao/pytorch_modules 
    git subtree add --prefix=<subtree_path> pytorch_modules master
    cd <subtree_path>
    pip install -r requirements.txt

### As a repository

    git clone https://github.com/woodsgao/pytorch_modules
    cd pytorch_modules
    pip install -r requirements.txt

## Usage

### pytorch_modules.nn

This module contains a variety of neural network layers, modules and loss functions.

    import torch
    from pytorch_modules.nn import ResBlock
    
    # NCHW tensor
    inputs = torch.ones([8, 8, 224, 224])
    block = ResBlock(8, 16)
    outputs = block(inputs)

### pytorch_modules.augments

See [woodsgao/image_augments](https://github.com/woodsgao/image_augments) for more details.

### pytorch_modules.backbones

This module includes a series of modified backbone networks, such as ResNet, SENet, Xception, DenseNet.

    import torch
    from pytorch_modules.backbones import ResNet
    
    # NCHW tensor
    inputs = torch.ones([8, 8, 224, 224])
    model = ResNet(32)
    outputs = model(inputs)

### pytorch_modules.datasets

This module includes a series of dataset classes integrated from `pytorch_modules.datasets.BasicDataset` which is integrated from `torch.utils.data.Dataset` .
The loading method of `pytorch_modules.datasets.BasicDataset` is modified to cache data to speed up data loading. This allows your gpu to be fully used for model training without spending a lot of time on data loading and data augmentation. You need to set parameter `cache_size` to use the cache function. It means the number of MBs the dataloader will occupy.
Please see the corresponding repository for detailed usage.

 - `pytorch_modules.datasets.ClassificationDataset` > [woodsgao/pytorch_classification](https://github.com/woodsgao/pytorch_classification)
 - `pytorch_modules.datasets.SegmentationDataset` > [woodsgao/pytorch_segmentation](https://github.com/woodsgao/pytorch_segmentation)

