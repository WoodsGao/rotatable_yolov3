import random
from pathlib import Path
import torch.distributed as dist
import cv2
#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

#matplotlib.rc('font', **{'size': 11})


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(
        None, names))  # filter removes empty strings (such as last line)


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    ni = len(labels)  # number of images
    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class

    # Prepend gridpoint count (for uCE trianing)
    gpi = ((320 / 32 * np.array([1, 2, 4]))**2 *
           3).sum()  # gridpoints per image
    weights = np.hstack([gpi * ni - weights.sum() * 9,
                         weights * 9])**0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class mAPs
    n = len(labels)
    class_counts = np.array([
        np.bincount(labels[i][:, 0].astype(np.int), minlength=nc)
        for i in range(n)
    ])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x,
                                          torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x,
                                          torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    coords[:, [0, 2]] *= img0_shape[1] / img1_shape[1]
    coords[:, [1, 3]] *= img0_shape[0] / img1_shape[0]
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0,
                                              max=img_shape[1])  # clip x
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0,
                                              max=img_shape[0])  # clip y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall = tpc / (n_gt + 1e-5)  # recall curve
            r.append(recall[-1])

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p.append(precision[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall, precision))

            # Plot
            # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            # ax.plot(np.concatenate(([0.], recall)), np.concatenate(([0.], precision)))
            # ax.set_xlabel('YOLOv3-SPP')
            # ax.set_xlabel('Recall')
            # ax.set_ylabel('Precision')
            # ax.set_xlim(0, 1)
            # fig.tight_layout()
            # fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-5)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(
            mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-5) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    iou = inter_area / union_area  # iou
    if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
        c_x1, c_x2 = torch.min(b1_x1, b2_x1), torch.max(b1_x2, b2_x2)
        c_y1, c_y2 = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + 1e-5  # convex area
        return iou - (c_area - union_area) / c_area  # GIoU

    return iou


def wh_iou(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-5) + w2 * h2 - inter_area

    return inter_area / union_area  # iou


def compute_loss(p, targets, model):  # predictions, targets, model
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    tcls, tbox, indices, anchor_vec = build_targets(model, targets)
    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss()
    BCEobj = nn.BCEWithLogitsLoss()
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()  # weight=model.class_weights
    # Compute losses
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj

        # Compute losses
        nb = len(b)
        if nb:  # number of targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            tobj[b, a, gj, gi] = 1.0  # obj
            # ps[:, 2:4] = torch.sigmoid(ps[:, 2:4])  # wh power loss (uncomment)

            # GIoU
            pxy = torch.sigmoid(
                ps[:, 0:2]
            )  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pbox = torch.cat((pxy, torch.exp(ps[:, 2:4]) * anchor_vec[i]),
                             1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False,
                            GIoU=True)  # giou computation
            lbox += (1.0 - giou).mean()  # giou loss

            # if model.nc > 1:  # cls loss (only if multiple classes)
            t = torch.zeros_like(ps[:, 5:])  # targets
            t[range(nb), tcls[i]] = 1.0
            lcls += BCE(ps[:, 5:], t).mean()  # BCE
            #ce = CE(ps[:, 5:], tcls[i])  # CE
            #lcls += ce
            # Instance-class weighting (use with reduction='none')
            # nt = t.sum(0) + 1  # number of targets per class
            # lcls += (BCEcls(ps[:, 5:], t) / nt).mean() * nt.mean()  # v1
            # lcls += (BCEcls(ps[:, 5:], t) / nt[tcls[i]].view(-1,1)).mean() * nt.mean()  # v2

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
        bce = BCE(pi[..., 4], tobj)
        lobj += bce  # obj loss

    lbox *= 3.31
    lobj *= 42.4
    lcls *= 40.
    loss = lbox + lobj + lcls
    return loss


def build_targets(model, targets):
    # targets = [image, class, x, y, w, h]
    if dist.is_initialized():
        yolo_layers = model.module.yolo_layers
    else:
        yolo_layers = model.yolo_layers
    nt = len(targets)
    tcls, tbox, indices, av = [], [], [], []
    for i in yolo_layers:
        # get number of grid points and anchor vec for this yolo layer
        ng, anchor_vec = i.ng, i.anchor_vec

        # iou of targets-anchors
        t, a = targets, []
        gwh = t[:, 4:6] * ng
        if nt:
            iou = torch.stack([wh_iou(x, gwh) for x in anchor_vec], 0)

            na = len(anchor_vec)  # number of anchors
            a = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1)
            t = targets.repeat([na, 1])
            gwh = gwh.repeat([na, 1])
            iou = iou.view(-1)  # use all ious

            # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
            j = iou > 0.213  # iou threshold hyperparameter
            t, a, gwh = t[j], a[j], gwh[j]

        # Indices
        b, c = t[:, :2].long().t()  # target image, class
        gxy = t[:, 2:4] * ng  # grid x, y
        gi, gj = gxy.long().t()  # grid x, y indices
        indices.append((b, a, gj, gi))

        # GIoU
        gxy -= gxy.floor()  # xy
        tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
        av.append(anchor_vec[a])  # anchor vec

        # Class
        tcls.append(c)

    return tcls, tbox, indices, av


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_conf, class)
    """

    min_wh = 2  # (pixels) minimum box width and height

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] =
        #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])

        # Multiply conf by class conf to get combined confidence
        class_conf, class_pred = pred[:, 5:].max(1)
        pred[:, 4] *= class_conf

        # # Merge classes (optional)
        # class_pred[(class_pred.view(-1,1) == torch.LongTensor([2, 3, 5, 6, 7]).view(1,-1)).any(1)] = 2
        #
        # # Remove classes (optional)
        # pred[class_pred != 2, 4] = 0.0

        # Select only suitable predictions
        i = (pred[:, 4] > conf_thres) & (
            pred[:, 2:4] > min_wh).all(1) & torch.isfinite(pred).all(1)
        pred = pred[i]

        # If none are remaining => process next image
        if len(pred) == 0:
            continue

        # Select predicted classes
        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        # pred[:, 4] *= class_conf  # improves mAP from 0.549 to 0.551

        # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
        pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)

        # Get detections sorted by decreasing confidence scores
        pred = pred[(-pred[:, 4]).argsort()]

        det_max = []
        nms_style = 'OR'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for c in pred[:, -1].unique():
            dc = pred[pred[:, -1] == c]  # select class c
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > 100:
                dc = dc[:
                        100]  # limit to first 100 boxes: https://github.com/ultralytics/yolov3/issues/117

            # Non-maximum suppression
            if nms_style == 'OR':  # default
                # METHOD1
                # ind = list(range(len(dc)))
                # while len(ind):
                # j = ind[0]
                # det_max.append(dc[j:j + 1])  # save highest conf detection
                # reject = (bbox_iou(dc[j], dc[ind]) > nms_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]

                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

            elif nms_style == 'SOFT':  # soft-NMS https://arxiv.org/abs/1704.04503
                sigma = 0.5  # soft-nms sigma parameter
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    det_max.append(dc[:1])
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:]
                    dc[:, 4] *= torch.exp(-iou**2 / sigma)  # decay confidences
                    # dc = dc[dc[:, 4] > nms_thres]  # new line per https://github.com/ultralytics/yolov3/issues/362

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort

    return output


def apply_classifier(x, model, img, im0):
    # applies a second stage classifier to yolo outputs

    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0.shape)

            # Classes
            pred_cls1 = d[:, 6].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0,
                                              1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im,
                                          dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(
                1)  # classifier prediction
            x[i] = x[i][pred_cls1 ==
                        pred_cls2]  # retain matching class detections

    return x


# Plotting functions ---------------------------------------------------------------------------------------------------
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)


def plot_images(imgs, targets, paths=None, fname='images.jpg'):
    # Plots training images overlaid with targets
    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()
    # targets = targets[targets[:, 1] == 21]  # plot only one class

    fig = plt.figure(figsize=(10, 10))
    bs, _, h, w = imgs.shape  # batch size, _, height, width
    bs = min(bs, 16)  # limit plot to 16 images
    ns = np.ceil(bs**0.5)  # number of subplots

    for i in range(bs):
        boxes = xywh2xyxy(targets[targets[:, 0] == i, 2:6]).T
        boxes[[0, 2]] *= w
        boxes[[1, 3]] *= h
        plt.subplot(ns, ns, i + 1).imshow(imgs[i].transpose(1, 2, 0))
        plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '.-')
        plt.axis('off')
        if paths is not None:
            s = Path(paths[i]).name
            plt.title(s[:min(len(s), 40)],
                      fontdict={'size': 8})  # limit to 40 characters
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close()
