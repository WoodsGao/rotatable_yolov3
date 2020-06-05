import argparse
import os
import os.path as osp

import torch
import torch.nn as nn

from pytorch2caffe import pytorch2caffe
from pytorch_modules.utils import fuse
from models import YOLOV3


def export2caffe(weights, num_classes, img_size):
    os.environ['MODEL_EXPORT'] = '1'
    model = YOLOV3(num_classes)
    weights = torch.load(weights, map_location='cpu')
    model.load_state_dict(weights['model'])
    model.eval()
    fuse(model)
    name = 'RYOLOV3'
    dummy_input = torch.ones([1, 3, img_size[1], img_size[0]])
    pytorch2caffe.trans_net(model, dummy_input, name)
    pytorch2caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch2caffe.save_caffemodel('{}.caffemodel'.format(name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str)
    parser.add_argument('-s',
                        '--img_size',
                        type=int,
                        nargs=2,
                        default=[416, 416])
    parser.add_argument('-nc', '--num-classes', type=int, default=21)
    opt = parser.parse_args()

    export2caffe(opt.weights, opt.num_classes, opt.img_size)
