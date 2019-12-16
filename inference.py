import argparse
from sys import platform
import os
import random
import torch
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from utils.models import YOLOV3  # set ONNX_EXPORT in models.py
from utils.utils import plot_one_box
from utils.detect import detect
from pytorch_modules.utils import device, IMG_EXT


def inference(source,
              output,
              weights,
              img_size=320,
              conf_thres=0.3,
              nms_thres=0.5,
              view_img=False):
    img_size = int(img_size // 32 * 32)
    # Initialize
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder
    out_txt = os.path.join(output, 'txt')
    os.makedirs(out_txt)  # make new output folder

    # Initialize model
    model = YOLOV3(80, (img_size, img_size))

    # Load weights
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

    # Run inference
    names = os.listdir(source)
    names.sort()
    for name in tqdm(names):
        im0 = cv2.imread(os.path.join(source, name))
        dets = detect(model, im0, (img_size, img_size), conf_thres, nms_thres)
        for det in dets:
            # Write results
            for *xyxy, conf, _, cls in det:
                with open(
                        os.path.join(out_txt,
                                     os.path.splitext(name)[0] + '.txt'),
                        'a') as f:
                    f.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                if view_img:  # Add bbox to image
                    label = '%d %.2f' % (int(cls), conf)
                    plot_one_box(xyxy,
                                 im0,
                                 label=label,
                                 color=colors[int(cls)])
            # Stream results
            if view_img:
                cv2.imshow('yolo', im0)
                cv2.waitKey(1)
            # Save results (image with detections)
            cv2.imwrite(os.path.join(output, name), im0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        type=str,
                        default='weights/best_mAP.pt',
                        help='path to weights file')
    parser.add_argument('--source',
                        type=str,
                        default='data/samples',
                        help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output',
                        type=str,
                        default='output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size',
                        type=int,
                        default=416,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.5,
                        help='object confidence threshold')
    parser.add_argument('--nms-thres',
                        type=float,
                        default=0.5,
                        help='iou threshold for non-maximum suppression')
    parser.add_argument('--view-img',
                        action='store_true',
                        help='display results')
    opt = parser.parse_args()
    print(opt)

    inference(
        opt.source,
        opt.output,
        opt.weights,
        img_size=opt.img_size,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        view_img=opt.view_img,
    )
