import os
import argparse
import torch
import torch.distributed as dist
from tqdm import tqdm
from test import test
from models import YOLOV3
from utils.utils import ComputeLoss
from utils.modules.utils import Trainer, Fetcher
from utils.modules.datasets import DetectionDataset
from torch.utils.data import DataLoader, DistributedSampler


def train(data_dir,
          epochs=100,
          img_size=224,
          batch_size=8,
          accumulate=2,
          lr=1e-3,
          adam=False,
          weights='',
          num_workers=0,
          augments={},
          multi_scale=False,
          notest=False,
          mixed_precision=False,
          local_rank=0):
    os.makedirs('weights', exist_ok=True)
    train_dir = os.path.join(data_dir, 'train.txt')
    val_dir = os.path.join(data_dir, 'valid.txt')
    train_data = DetectionDataset(
        train_dir,
        img_size=img_size,
        augments=augments,
        skip_init=(local_rank > 0),
    )
    if not notest:
        val_data = DetectionDataset(
            val_dir,
            img_size=img_size,
            augments={},
            skip_init=(local_rank > 0),
        )
    if dist.is_initialized():
        dist.barrier()
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=not (dist.is_initialized()),
        sampler=DistributedSampler(train_data, dist.get_world_size(),
                                   dist.get_rank())
        if dist.is_initialized() else None,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=DetectionDataset.collate_fn, 
    )
    train_fetcher = Fetcher(train_loader, train_data.post_fetch_fn)
    if not notest:
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=not (dist.is_initialized()),
            sampler=DistributedSampler(val_data, dist.get_world_size(),
                                       dist.get_rank())
            if dist.is_initialized() else None,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=DetectionDataset.collate_fn, 
        )
        val_fetcher = Fetcher(val_loader, post_fetch_fn=val_data.post_fetch_fn)
    model = YOLOV3(80)
    # maps = np.zeros()  # mAP per class

    compute_loss = ComputeLoss(model)
    trainer = Trainer(model, train_fetcher, compute_loss, weights, accumulate,
                      adam, lr, mixed_precision)
    while trainer.epoch < epochs:
        trainer.run_epoch()
        save_path_list = ['last.pt']
        if trainer.epoch % 10 == 0:
            save_path_list.append('bak%d.pt' % trainer.epoch)
        if not notest:
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
        save_path_list = [os.path.join('weights', p) for p in save_path_list]
        if local_rank > 0:
            continue
        trainer.save(save_path_list)


if __name__ == "__main__":
    if dist.is_available():
        try:
            dist.init_process_group(backend="gloo", init_method="env://")
        except:
            pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/voc')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulate', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument('--adam', action='store_true')
    parser.add_argument('--mp', action='store_true', help='mixed precision')
    parser.add_argument('--notest', action='store_true')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    augments = {
        'hsv': 0.1,
        'blur': 0.1,
        'pepper': 0.1,
        'shear': 0.1,
        'translate': 0.1,
        'rotate': 0.1,
        'flip': 0.1,
        'scale': 0.1,
        'noise': 0.1,
    }
    opt = parser.parse_args()
    print(opt)
    train(
        data_dir=opt.data_dir,
        epochs=opt.epochs,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        accumulate=opt.accumulate,
        lr=opt.lr,
        weights=opt.weights,
        num_workers=opt.num_workers,
        augments=augments,
        multi_scale=opt.multi_scale,
        notest=opt.notest,
        adam=opt.adam,
        mixed_precision=opt.mp,
        local_rank=opt.local_rank,
    )
    if dist.is_initialized():
        dist.destroy_process_group()
