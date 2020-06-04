import argparse
import os.path as osp

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import YOLOV3
from pytorch_modules.utils import Fetcher, device
from utils.datasets import CocoDataset
from utils.utils import (ap_per_class, bbox_iou, clip_coords, compute_loss,
                         non_max_suppression, polygon_iou, show_batch,
                         xywh2xyxy, xywht2polygon)


@torch.no_grad()
def test(model, fetcher, conf_thres=1e-3, nms_thres=0.5):
    model.eval()
    val_loss = 0
    classes = fetcher.loader.dataset.classes
    num_classes = len(classes)
    seen = 0
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP',
                                 'F1')
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(enumerate(fetcher), total=len(fetcher))
    for idx, (imgs, targets) in pbar:
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        # Run model
        inf_out, train_out = model(imgs)  # inference and training outputs

        # Compute loss
        val_loss += compute_loss(train_out, targets,
                                 model).item()  # GIoU, obj, cls
        nb, nbi, _ = inf_out.shape
        polygons = xywht2polygon(
            torch.cat([inf_out[..., :4], inf_out[..., -1:]],
                      2).reshape(-1, 5).t()).view(nb, nbi, 8)
        conf, cls_idx = inf_out[..., 5:-1].max(2)
        conf *= inf_out[..., 4]
        inf_out = torch.cat(
            [polygons, torch.stack([conf, cls_idx.float()], 2)], 2)
        # Run NMS
        output = non_max_suppression(inf_out,
                                     conf_thres=conf_thres,
                                     nms_thres=nms_thres)
        # Plot images with bounding boxes
        if idx == 0:
            show_batch(imgs, output)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Clip boxes to image bounds
            # clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]
                # target
                tpoly = xywht2polygon(labels[:, 1:].t())
                tpoly[:, 0:8:2] *= width
                tpoly[:, 1:8:2] *= height

                # Search for correct predictions
                for i, (*ppoly, pconf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue
                    ppoly = torch.FloatTensor(ppoly).to(pred.device)
                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = polygon_iou(ppoly, tpoly[m]).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > 0.5 and m[
                            bi] not in detected:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append(m[bi])

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, -2].cpu(), pred[:, -1].cpu(), tcls))
        pbar.set_description('loss: %8g' % (val_loss / (idx + 1)))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]

    # sync stats
    if dist.is_initialized():
        for i in range(len(stats)):
            stat = torch.FloatTensor(stats[i]).to(device)
            ls = torch.IntTensor([len(stat)]).to(device)
            ls_list = [
                torch.IntTensor([0]).to(device)
                for _ in range(dist.get_world_size())
            ]
            dist.all_gather(ls_list, ls)
            ls_list = [ls_item.item() for ls_item in ls_list]
            max_ls = max(ls_list)
            if len(stat) < max_ls:
                stat = torch.cat(
                    [stat, torch.zeros(max_ls - len(stat)).to(device)])
            stat_list = [
                torch.zeros(max_ls).to(device)
                for _ in range(dist.get_world_size())
            ]
            dist.all_gather(stat_list, stat)
            stat_list = [
                stat_list[si][:ls_list[si]]
                for si in range(dist.get_world_size()) if ls_list[si] > 0
            ]
            if len(stat_list) > 0:
                stat = torch.cat(stat_list)
                stats[i] = stat.cpu().numpy()

    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64),
                         minlength=num_classes)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    for i, c in enumerate(ap_class):
        print(pf % (classes[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
    # Return results
    maps = np.zeros(num_classes) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    # return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist()), maps
    print(val_loss / len(fetcher))
    return map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-list', type=str, default='data/voc/valid.txt')
    parser.add_argument('--img-size', type=str, default='512')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.1,
                        help='object confidence threshold')
    parser.add_argument('--nms-thres',
                        type=float,
                        default=0.5,
                        help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()

    img_size = opt.img_size.split(',')
    assert len(img_size) in [1, 2]
    if len(img_size) == 1:
        img_size = [int(img_size[0])] * 2
    else:
        img_size = [int(x) for x in img_size]

    val_data = CocoDataset(opt.val_list, img_size=tuple(img_size), augment=None)
    val_loader = DataLoader(
        val_data,
        batch_size=opt.batch_size,
        pin_memory=True,
        num_workers=opt.num_workers,
        collate_fn=CocoDataset.collate_fn,
    )
    val_fetcher = Fetcher(val_loader, post_fetch_fn=val_data.post_fetch_fn)
    model = YOLOV3(len(val_data.classes))
    if opt.weights:
        state_dict = torch.load(opt.weights, map_location='cpu')
        model.load_state_dict(state_dict['model'])
    metrics = test(model,
                   val_fetcher,
                   conf_thres=opt.conf_thres,
                   nms_thres=opt.nms_thres)
    print('metrics: %8g' % (metrics))
