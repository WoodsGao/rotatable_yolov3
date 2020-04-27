import os
import os.path as osp
import argparse
import torch
import torch.distributed as dist
from tqdm import tqdm
from test import test
from utils.models import YOLOV3
from utils.utils import compute_loss
from utils.datasets import CocoDataset
from pytorch_modules.utils import Trainer, Fetcher
from torch.utils.data import DataLoader, DistributedSampler


def train(data_dir,
          epochs=100,
          img_size=(416, 416),
          batch_size=8,
          accumulate=2,
          lr=1e-3,
          adam=False,
          weights='',
          num_workers=0,
          multi_scale=False,
          rect=False,
          notest=False,
          mixed_precision=False,
          nosave=False):
    os.makedirs('weights', exist_ok=True)
    train_coco = osp.join(data_dir, 'train.json')
    val_coco = osp.join(data_dir, 'val.json')

    train_data = CocoDataset(train_coco, img_size=img_size,
                             multi_scale=multi_scale, rect=rect)
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
        val_data = CocoDataset(val_coco, img_size=img_size, augments=None, rect=rect)
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

    trainer = Trainer(model, train_fetcher, loss_fn=compute_loss, weights=weights, accumulate=accumulate,
                      adam=adam, lr=lr, mixed_precision=mixed_precision)
    while trainer.epoch < epochs:
        trainer.step()
        save_path_list = ['last.pt']
        if trainer.epoch % 10 == 0:
            save_path_list.append('bak%d.pt' % trainer.epoch)
        if not notest and trainer.epoch >= 10:
            metrics = test(
                trainer.model,
                val_fetcher,
                conf_thres=0.001
                if trainer.epoch > 20 else 0.1,  # 0.1 for speed
            )
            if metrics > trainer.metrics:
                trainer.metrics = metrics
                save_path_list.append('best.pt')
                print('save best, metrics: %g...' % metrics)
        save_path_list = [osp.join('weights', p) for p in save_path_list]
        if nosave:
            continue
        trainer.save(save_path_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/voc')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--img-size', type=str, default='416')
    parser.add_argument('-bs', '--batch-size', type=int, default=4)
    parser.add_argument('-a', '--accumulate', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adam', action='store_true')
    parser.add_argument('--mp', action='store_true', help='mixed precision')
    parser.add_argument('--notest', action='store_true')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--rect', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    opt = parser.parse_args()
    print(opt)

    if torch.distributed.is_available() and os.environ.get('WORLD_SIZE'):
        torch.distributed.init_process_group(backend=opt.backend,
                                             init_method='env://',
                                             world_size=int(
                                                 os.environ['WORLD_SIZE']),
                                             rank=int(os.environ['RANK']))
    torch.cuda.set_device(opt.local_rank)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(opt.local_rank)
    if opt.local_rank > 0:
        sys.stdout = open(os.devnull, 'w')
        print('NB')
    print(opt)
    img_size = opt.img_size.split(',')
    assert len(img_size) in [1, 2]
    if len(img_size) == 1:
        img_size = [int(img_size[0])] * 2
    else:
        img_size = [int(x) for x in img_size]

    train(
        data_dir=opt.data,
        epochs=opt.epochs,
        img_size=tuple(img_size),
        batch_size=opt.batch_size,
        accumulate=opt.accumulate,
        lr=opt.lr,
        weights=opt.weights,
        num_workers=opt.num_workers,
        multi_scale=opt.multi_scale,
        rect=opt.rect,
        notest=opt.notest,
        adam=opt.adam,
        mixed_precision=opt.mp,
        nosave=opt.nosave or (opt.local_rank > 0),
    )
    if dist.is_initialized():
        dist.destroy_process_group()
