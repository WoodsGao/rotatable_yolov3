import argparse
import os
import os.path as osp
import random
import shutil

import cv2
import torch
from tqdm import tqdm

from models import YOLOV3  # set ONNX_EXPORT in models.py
from pytorch_modules.utils import IMG_EXT, device
from utils.inference import inference
from utils.utils import plot_one_box


def run(img_dir,
        output_dir,
        img_size,
        num_classes,
        weights,
        conf_thres,
        nms_thres,
        show):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    model = YOLOV3(num_classes, img_size)
    state_dict = torch.load(weights, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model = model.to(device)
    model.eval()
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]
    names = [n for n in os.listdir(img_dir) if osp.splitext(n)[1] in IMG_EXT]
    names.sort()
    for name in tqdm(names):
        img = cv2.imread(osp.join(img_dir, name))
        det = inference(model, [img], img_size, conf_thres,
                        nms_thres)[0]
        det_txt = []
        # Write results
        for *xyxy, conf, _, cls in det:
            det_txt.append(' '.join(['%g'] * 6) % (*xyxy, cls, conf))
            if show:  # Add bbox to image
                label = '%d %.2f' % (int(cls), conf)
                plot_one_box(xyxy, img, label=label, color=colors[int(cls)])
        with open(osp.join(output_dir,
                           osp.splitext(name)[0] + '.txt'), 'w') as f:
            f.write('\n'.join(det_txt))
        # Stream results
        if show:
            cv2.imshow('yolo', img)
            cv2.waitKey(1)
        # Save results (image with detections)
        cv2.imwrite(osp.join(output_dir, name), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('-s',
                        '--img_size',
                        type=int,
                        nargs=2,
                        default=[416, 416])
    parser.add_argument('-nc', '--num-classes', type=int, default=21)
    parser.add_argument('--weights', type=str, default='weights/best.pt')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.5,
                        help='object confidence threshold')
    parser.add_argument('--nms-thres',
                        type=float,
                        default=0.5,
                        help='iou threshold for non-maximum suppression')
    parser.add_argument('--show', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    run(opt.img_dir, opt.output_dir, opt.img_size, opt.num_classes, opt.weights,
        opt.conf_thres, opt.nms_thres, opt.show)
