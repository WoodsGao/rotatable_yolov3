import argparse
import torch
import numpy as np
from tqdm import tqdm
from models import YOLOV3
from utils.utils import compute_loss, non_max_suppression, clip_coords, xywh2xyxy, bbox_iou, ap_per_class


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
    with torch.no_grad():
        pbar = tqdm(enumerate(fetcher), total=len(fetcher))
        for batch_i, (imgs, targets) in pbar:
            _, _, height, width = imgs.shape  # batch size, channels, height, width

            # Plot images with bounding boxes
            # if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
            #     plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')

            # Run model
            inf_out, train_out = model(imgs)  # inference and training outputs

            # Compute loss
            val_loss += compute_loss(train_out,
                                     targets, model).item()  # GIoU, obj, cls

            # Run NMS
            output = non_max_suppression(inf_out,
                                         conf_thres=conf_thres,
                                         nms_thres=nms_thres)

            # Statistics per image
            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if pred is None:
                    if nl:
                        stats.append(
                            ([], torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Append to text file
                # with open('test.txt', 'a') as file:
                #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

                # Clip boxes to image bounds
                clip_coords(pred, (height, width))

                # Assign all predictions as incorrect
                correct = [0] * len(pred)
                if nl:
                    detected = []
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    tbox[:, [0, 2]] *= width
                    tbox[:, [1, 3]] *= height

                    # Search for correct predictions
                    for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                        # Break if all targets already located in image
                        if len(detected) == nl:
                            break

                        # Continue if predicted class not among image classes
                        if pcls.item() not in tcls:
                            continue

                        # Best iou, index between pred and targets
                        m = (pcls == tcls_tensor).nonzero().view(-1)
                        iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                        # If iou > threshold and class is correct mark as correct
                        if iou > 0.5 and m[
                                bi] not in detected:  # and pcls == tcls[bi]:
                            correct[i] = 1
                            detected.append(m[bi])

                # Append statistics (correct, conf, pcls, tcls)
                stats.append(
                    (correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
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
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg',
                        type=str,
                        default='cfg/yolov3-spp.cfg',
                        help='cfg file path')
    parser.add_argument('--data',
                        type=str,
                        default='data/coco.data',
                        help='coco.data file path')
    parser.add_argument('--weights',
                        type=str,
                        default='weights/yolov3-spp.weights',
                        help='path to weights file')
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='size of each image batch')
    parser.add_argument('--img-size',
                        type=int,
                        default=416,
                        help='inference size (pixels)')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.5,
                        help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.001,
                        help='object confidence threshold')
    parser.add_argument('--nms-thres',
                        type=float,
                        default=0.5,
                        help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json',
                        action='store_true',
                        help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--device',
                        default='',
                        help='device id (i.e. 0 or 0,1) or cpu')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        test(opt.cfg, opt.data, opt.weights, opt.batch_size, opt.img_size,
             opt.iou_thres, opt.conf_thres, opt.nms_thres, opt.save_json)
