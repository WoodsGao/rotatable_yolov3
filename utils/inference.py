import numpy as np
import cv2
import torch
from pytorch_modules.utils import device
from .utils import non_max_suppression, scale_coords


@torch.no_grad()
def inference(model, imgs, img_size=(320, 320), conf_thres=0.3, nms_thres=0.5):
    shapes = [img.shape for img in imgs]
    imgs = [
        cv2.resize(img, img_size)[:, :, ::-1].transpose(2, 0, 1)
        for img in imgs
    ]
    imgs = torch.FloatTensor(imgs).to(device) / 255.
    preds = model(imgs)

    dets = []
    for pred, shape, img in zip(preds, shapes, imgs):
        # Apply NMS
        det = non_max_suppression(pred, conf_thres, nms_thres)[0]

        # Process detections
        if det is None:
            det = []
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                      shape[:2]).round()
        dets.append(det)
    return dets