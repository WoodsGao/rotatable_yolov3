import argparse
import os
import os.path as osp
import random
import torch
import cv2
import shutil
from tqdm import tqdm
from models import YOLOV3  # set ONNX_EXPORT in models.py
from utils.utils import plot_one_box
from utils.inference import inference
from pytorch_modules.utils import device, IMG_EXT


def run(img_dir='data/samples',
        img_size=(416, 416),
        num_classes=80,
        output_dir='outputs',
        weights='weights/best.pt',
        conf_thres=0.3,
        nms_thres=0.5,
        show=False):
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
    parser.add_argument('--src', type=str, default='data/samples')
    parser.add_argument('--dst', type=str, default='outputs')
    parser.add_argument('--img-size', type=str, default='416')
    parser.add_argument('--num-classes', type=int, default=80)
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

    img_size = opt.img_size.split(',')
    assert len(img_size) in [1, 2]
    if len(img_size) == 1:
        img_size = [int(img_size[0]), int(img_size[0])]
    else:
        img_size = [int(x) for x in img_size]

    run(opt.src, tuple(img_size), opt.num_classes, opt.dst, opt.weights,
        opt.conf_thres, opt.nms_thres, opt.show)
