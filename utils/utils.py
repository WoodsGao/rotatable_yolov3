import math
import random

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from shapely.geometry import MultiPoint, Polygon

from pytorch_modules.nn import FocalBCELoss


def angle_loss(x, period=math.pi):
    x %= period
    x[x > period / 2.] *= -1
    x[x < 0] += period
    return x


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
    coords[:, 0:8:2] *= img0_shape[1] / img1_shape[1]
    coords[:, 1:8:2] *= img0_shape[0] / img1_shape[0]
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(poly, img_shape):
    poly[:, 0:8:2] = poly[:, 0:8:2].clamp(min=0,
                                              max=img_shape[1])  # clip x
    poly[:, 1:8:2] = poly[:, 1:8:2].clamp(min=0,
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


def polygon_iou(polygon1, polygon2):
    iou = torch.zeros(len(polygon2)).to(polygon2.device)
    poly1 = Polygon(polygon1.cpu().numpy()[:8].reshape(4, 2)).convex_hull
    for i, poly in enumerate(polygon2):
        poly2 = Polygon(poly.cpu().numpy()[:8].reshape(4, 2)).convex_hull
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        iou[i] = inter_area / (union_area + 1e-5)
    return iou


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


def hungary(task_matrix):
    task_matrix = task_matrix.copy()
    # 行和列减0
    nb, nr, nc = task_matrix.shape
    task_matrix -= task_matrix.min(1, keepdims=True)
    task_matrix -= task_matrix.min(2, keepdims=True)
    solutions = []
    for b in task_matrix:
        line_count = 0
        # 线数目小于矩阵长度时，进行循环
        while (line_count < len(b)):
            line_count = 0
            row_zero_count = []
            col_zero_count = []
            for i in range(len(b)):
                row_zero_count.append(np.sum(b[i] == 0))
            for i in range(len(b[0])):
                col_zero_count.append((np.sum(b[:, i] == 0)))
            # 划线的顺序（分行或列）
            line_order = []
            row_or_col = []
            for i in range(len(b[0]), 0, -1):
                while (i in row_zero_count):
                    line_order.append(row_zero_count.index(i))
                    row_or_col.append(0)
                    row_zero_count[row_zero_count.index(i)] = 0
                while (i in col_zero_count):
                    line_order.append(col_zero_count.index(i))
                    row_or_col.append(1)
                    col_zero_count[col_zero_count.index(i)] = 0
            # 画线覆盖0，并得到行减最小值，列加最小值后的矩阵
            delete_count_of_row = []
            delete_count_of_rol = []
            row_and_col = [i for i in range(len(b))]
            for i in range(len(line_order)):
                if row_or_col[i] == 0:
                    delete_count_of_row.append(line_order[i])
                else:
                    delete_count_of_rol.append(line_order[i])
                c = np.delete(b, delete_count_of_row, axis=0)
                c = np.delete(c, delete_count_of_rol, axis=1)
                line_count = len(delete_count_of_row) + len(
                    delete_count_of_rol)
                # 线数目等于矩阵长度时，跳出
                if line_count == len(b):
                    break
                # 判断是否画线覆盖所有0，若覆盖，进行加减操作
                if 0 not in c:
                    row_sub = list(set(row_and_col) - set(delete_count_of_row))
                    min_value = np.min(c)
                    for i in row_sub:
                        b[i] = b[i] - min_value
                    for i in delete_count_of_rol:
                        b[:, i] = b[:, i] + min_value
                    break
        row_ind, col_ind = linear_sum_assignment(b)
        # best_solutions = list(task_matrix[row_ind, col_ind])
        # print(row_ind, col_ind)
        solution = np.zeros_like(task_matrix[0])
        inds = np.stack([row_ind, col_ind], 1)
        for ind in inds:
            solution[tuple(ind)] = 1
        solutions.append(solution)
    solutions = np.stack(solutions, 0)
    return solutions


def xywht2polygon(rbox):
    x, y, w, h, theta = rbox
    w /= 2.
    h /= 2.
    p1 = torch.stack([
        x - torch.sin(theta) * h - torch.cos(theta) * w,
        y - torch.sin(theta) * w + torch.cos(theta) * h
    ], 1)
    p2 = torch.stack([
        x + torch.sin(theta) * h - torch.cos(theta) * w,
        y - torch.sin(theta) * w - torch.cos(theta) * h
    ], 1)
    p3 = torch.stack([
        x + torch.sin(theta) * h + torch.cos(theta) * w,
        y + torch.sin(theta) * w - torch.cos(theta) * h
    ], 1)
    p4 = torch.stack([
        x - torch.sin(theta) * h + torch.cos(theta) * w,
        y + torch.sin(theta) * w + torch.cos(theta) * h
    ], 1)
    points = torch.stack([p1, p2, p3, p4], 2).permute(0, 2, 1).contiguous()
    return points


def compute_loss(p, targets, model):  # predictions, targets, model
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    lcls, lbox, lobj, lt = ft([0]), ft([0]), ft([0]), ft([0])
    tcls, tbox, indices, anchor_vec, ttheta = build_targets(model, targets)
    # Define criteria
    BCE = nn.BCELoss()
    BCE = FocalBCELoss()
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
            pxy = torch.sigmoid(ps[:, 0:2])
            pbox = torch.cat(
                (pxy, ps[:, 2:4].exp() * anchor_vec[i]),
                1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False,
                            GIoU=True)  # giou computation
            lbox += (1.0 - giou).mean()  # giou loss

            # radius = w^2+h^2
            # 长宽中找到最近的一点 也就是计算-cos(theta)和-cos(theta_wh-theta)和-cos(pi-theta_wh-theta)的最小值
            l2 = angle_loss(ps[:, -1] - ttheta[i])
            # true_points = xywht2polygon(
            #     torch.cat([tbox[i], ttheta[i].unsqueeze(1)], 1).t())
            # pred_points = xywht2polygon(
            #     torch.cat([tbox[i], ps[:, -1:]], 1).t())
            # # print(true_points[0], pred_points.unsqueeze(1).repeat(1, 4, 1, 1).permute(0, 2, 1, 3)[0])
            # l2 = (pred_points.unsqueeze(1).repeat(1, 4, 1, 1).permute(
            #     0, 2, 1, 3) - true_points.unsqueeze(1))
            # l2 = l2[..., 0]**2 + l2[..., 1]**2
            # solution = hungary(l2.cpu().detach().numpy())  # + np.eye(4) * 1
            # solution = ft(solution)
            # l2 *= solution
            # l2 = l2.sum(1).sum(1)
            lt += l2.mean()
            # print(l2.shape)
            t = torch.zeros_like(ps[:, 5:-1])  # targets
            t[range(nb), tcls[i]] = 1.0
            lcls += BCE(ps[:, 5:-1].sigmoid(), t).mean()  # BCE

        bce = BCE(pi[..., 4].sigmoid(), tobj)
        lobj += bce  # obj loss

    lbox *= 3.54
    lobj *= 64.3
    lcls *= 37.4
    # lt = torch.cat(lt)
    # if len(lt) > 100:
    #     lt = lt.topk(100)[0]
    # lt = lt.mean()
    # print(lt)
    lt *= 10
    loss = lbox + lobj + lcls + lt
    # loss = lt
    return loss


def build_targets(model, targets):
    # targets = [image, class, x, y, w, h]
    if dist.is_initialized():
        yolo_layers = model.module.yolo_layers
    else:
        yolo_layers = model.yolo_layers
    nt = len(targets)
    tcls, tbox, indices, av, ttheta = [], [], [], [], []
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
        ttheta.append(t[:, 6].float())
        gxy = t[:, 2:4] * ng  # grid x, y
        gi, gj = gxy.long().t()  # grid x, y indices
        indices.append((b, a, gj, gi))

        # GIoU
        gxy -= gxy.floor()  # xy
        tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
        av.append(anchor_vec[a])  # anchor vec

        # Class
        tcls.append(c)

    return tcls, tbox, indices, av, ttheta


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, x3, y3, x4, y4, conf, class)
    """

    output = [torch.zeros((0, 10)).to(prediction[0].device)] * len(prediction)
    for image_i, pred in enumerate(prediction):
        i = (pred[:, 8] > conf_thres) & torch.isfinite(pred).all(1)
        pred = pred[i]

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
                # reject = (polygon_iou(dc[j], dc[ind]) > nms_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]

                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = polygon_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = polygon_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = polygon_iou(dc[0], dc) > nms_thres  # iou with other boxes
                    weights = dc[i, 8:9]
                    dc[0, :8] = (weights * dc[i, :8]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

            elif nms_style == 'SOFT':  # soft-NMS https://arxiv.org/abs/1704.04503
                sigma = 0.5  # soft-nms sigma parameter
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    det_max.append(dc[:1])
                    iou = polygon_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:]
                    dc[:, 8] *= torch.exp(-iou**2 / sigma)  # decay confidences
                    # dc = dc[dc[:, 8] > nms_thres]  # new line per https://github.com/ultralytics/yolov3/issues/362

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 8]).argsort()]  # sort
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
def show_target(inputs, targets):
    imgs = inputs.clone()[:8]
    bboxes = targets
    imgs *= torch.FloatTensor([58.395, 57.12,
                               57.375]).reshape(1, 3, 1, 1).to(imgs.device)
    imgs += torch.FloatTensor([123.675, 116.28,
                               103.53]).reshape(1, 3, 1, 1).to(imgs.device)

    imgs = imgs.clamp(0, 255).permute(0, 2, 3,
                                      1).byte().cpu().numpy()[..., ::-1]
    imgs = np.ascontiguousarray(imgs)
    for i in range(len(bboxes)):
        ii = int(bboxes[i, 0])
        if ii >= len(imgs):
            continue
        img = imgs[ii]
        xywh = bboxes[i:i + 1, 2:]
        xywh[:, (0, 2)] *= img.shape[1]
        xywh[:, (1, 3)] *= img.shape[0]
        # xyxy = xywh2xyxy(xywh)
        plot_one_poly(xyxy[0], img)
        imgs[ii] = img
    imgs = imgs.reshape(-1, imgs.shape[2], imgs.shape[3])

    save_img = imgs
    cv2.imwrite('batch.png', save_img)


def show_batch(inputs, targets):
    imgs = inputs.clone()[:8]
    polys = targets[:8]
    imgs *= torch.FloatTensor([58.395, 57.12,
                               57.375]).reshape(1, 3, 1, 1).to(imgs.device)
    imgs += torch.FloatTensor([123.675, 116.28,
                               103.53]).reshape(1, 3, 1, 1).to(imgs.device)

    imgs = imgs.clamp(0, 255).permute(0, 2, 3,
                                      1).byte().cpu().numpy()[..., ::-1]
    imgs = np.ascontiguousarray(imgs)
    for i in range(len(polys)):
        poly = polys[i].cpu().numpy()
        poly = poly[poly[..., 8] > 0.3]
        img = imgs[i]
        # clip_coords(poly, img.shape)
        for *xyxy, conf, c in poly:
            label = '%d %lf' % (c, conf)
            xyxy = np.int32(xyxy)
            plot_one_poly(xyxy, img, label=label)
        imgs[i] = img
    imgs = imgs.reshape(-1, imgs.shape[2], imgs.shape[3])

    save_img = imgs
    cv2.imwrite('batch.png', save_img)


def plot_one_poly(poly, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    cv2.drawContours(img, [poly.reshape(-1, 1, 2)], 0, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c1 = (int(poly[0]), int(poly[1]))
        c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
