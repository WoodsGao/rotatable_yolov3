import argparse
import os
import os.path as osp
import sys
from test import test

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from models import YOLOV3
from pytorch_modules.utils import Fetcher, Trainer
from utils.datasets import CocoDataset
from utils.utils import compute_loss


def train(data_dir,
          epochs=100,
          img_size=(416, 416),
          batch_size=8,
          accumulate=2,
          lr=1e-3,
          adam=False,
          resume=False,
          weights='',
          num_workers=0,
          multi_scale=False,
          rect=False,
          mixed_precision=False,
          notest=False,
          nosave=False):
    train_coco = osp.join(data_dir, 'train.json')
    val_coco = osp.join(data_dir, 'val.json')

    train_data = CocoDataset(train_coco,
                             img_size=img_size,
                             multi_scale=multi_scale,
                             rect=rect,
                             with_label=False,
                             mosaic=True)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=not (dist.is_initialized()),
        sampler=DistributedSampler(train_data, dist.get_world_size(),
                                   dist.get_rank())
        if dist.is_initialized() else None,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=CocoDataset.collate_fn,
    )
    train_fetcher = Fetcher(train_loader, train_data.post_fetch_fn)
    if not notest:
        val_data = CocoDataset(val_coco,
                               img_size=img_size,
                               augments=None,
                               rect=rect)
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=not (dist.is_initialized()),
            sampler=DistributedSampler(val_data, dist.get_world_size(),
                                       dist.get_rank())
            if dist.is_initialized() else None,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=CocoDataset.collate_fn,
        )
        val_fetcher = Fetcher(val_loader, post_fetch_fn=val_data.post_fetch_fn)

    model = YOLOV3(len(train_data.classes))

    trainer = Trainer(model,
                      train_fetcher,
                      loss_fn=compute_loss,
                      workdir='weights',
                      accumulate=accumulate,
                      adam=adam,
                      lr=lr,
                      weights=weights,
                      resume=resume,
                      mixed_precision=mixed_precision)
    while trainer.epoch < epochs:
        trainer.step()
        best = False
        if not notest:
            metrics = test(trainer.model, val_fetcher, conf_thres=0.1)
            if metrics > trainer.metrics:
                best = True
                print('save best, mAP: %g' % metrics)
                trainer.metrics = metrics
        if not nosave:
            trainer.save(best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/voc')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--img-size', type=str, default='416')
    parser.add_argument('-bs', '--batch-size', type=int, default=4)
    parser.add_argument('-a', '--accumulate', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adam', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--rect', action='store_true')
    parser.add_argument('-mp',
                        '--mix_precision',
                        action='store_true',
                        help='mixed precision')
    parser.add_argument('--notest', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    opt = parser.parse_args()

    if torch.distributed.is_available() and os.environ.get('WORLD_SIZE'):
        torch.distributed.init_process_group(backend=opt.backend,
                                             init_method='env://',
                                             world_size=int(
                                                 os.environ['WORLD_SIZE']),
                                             rank=int(os.environ['RANK']))
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.local_rank)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(opt.local_rank)
    if opt.local_rank > 0:
        sys.stdout = open(os.devnull, 'w')
    print(opt)
    img_size = opt.img_size.split(',')
    assert len(img_size) in [1, 2]
    if len(img_size) == 1:
        img_size = [int(img_size[0]), int(img_size[0])]
    else:
        img_size = [int(x) for x in img_size]

    train(data_dir=opt.data,
          epochs=opt.epochs,
          img_size=img_size,
          batch_size=opt.batch_size,
          accumulate=opt.accumulate,
          lr=opt.lr,
          adam=opt.adam,
          resume=opt.resume,
          weights=opt.weights,
          num_workers=opt.num_workers,
          multi_scale=opt.multi_scale,
          rect=opt.rect,
          mixed_precision=opt.mix_precision,
          notest=opt.notest,
          nosave=opt.nosave)
    if dist.is_initialized():
        dist.destroy_process_group()
