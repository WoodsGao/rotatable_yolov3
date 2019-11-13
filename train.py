import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from tqdm import tqdm
import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from utils.modules.optims import AdaBoundW
from utils.modules.datasets import DetectionDataset
from torch.utils.data import DataLoader

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

# Hyperparameters (k-series, 53.3 mAP yolov3-spp-320) https://github.com/ultralytics/yolov3/issues/310
hyp = {
    'giou': 3.31,  # giou loss gain
    'cls': 42.4,  # cls loss gain
    'cls_pw': 1.0,  # cls BCELoss positive_weight
    'obj': 40.0,  # obj loss gain
    'obj_pw': 1.0,  # obj BCELoss positive_weight
    'iou_t': 0.213,  # iou training threshold
    'lr0': 0.00261,  # initial learning rate (SGD=1E-3, Adam=9E-5)
    'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
    'momentum': 0.949,  # SGD momentum
    'weight_decay': 0.000489,  # optimizer weight decay
    'fl_gamma': 0.5,  # focal loss gamma
    'hsv_h': 0.0103,  # image HSV-Hue augmentation (fraction)
    'hsv_s': 0.691,  # image HSV-Saturation augmentation (fraction)
    'hsv_v': 0.433,  # image HSV-Value augmentation (fraction)
    'degrees': 1.43,  # image rotation (+/- deg)
    'translate': 0.0663,  # image translation (+/- fraction)
    'scale': 0.11,  # image scale (+/- gain)
    'shear': 0.384
}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

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


def train(lr=1e-3):
    cfg = opt.cfg
    data = opt.data
    img_size = opt.img_size
    epochs = 1 if opt.prebias else opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial training weights
    if 'pw' not in opt.arc:  # remove BCELoss positive weights
        hyp['cls_pw'] = 1.
        hyp['obj_pw'] = 1.

    # Initialize
    multi_scale = opt.multi_scale

    if multi_scale:
        img_sz_min = round(img_size / 32 / 1.5) + 1
        img_sz_max = round(img_size / 32 * 1.5) - 1
        img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

    # Configure run
    data_dict = parse_data_cfg(data)
    train_list = data_dict['train']
    val_list = data_dict['valid']
    nc = int(data_dict['classes'])  # number of classes

    # Initialize model
    model = YOLOV3(80).to(device)

    if opt.adam:
        optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            #   weight_decay=5e-4,
            nesterov=True)
    epoch = 0
    best_mAP = 0
    best_loss = 1000
    if opt.resume:
        state_dict = torch.load(weights, map_location=device)
        if opt.adam:
            if 'adam' in state_dict:
                optimizer.load_state_dict(state_dict['adam'])
        best_mAP = state_dict['mAP']
        best_loss = state_dict['loss']
        epoch = state_dict['epoch']
        model.load_state_dict(state_dict['model'], strict=True)
    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    # lf = lambda x: 1 - x / epochs  # linear ramp to zero
    # lf = lambda x: 10 ** (hyp['lrf'] * x / epochs)  # exp ramp
    # lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inverse exp ramp
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=range(59, 70, 1), gamma=0.8)  # gradual fall to 0.1*lr0
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    # scheduler.last_epoch = start_epoch - 1

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level='O1',
                                          verbosity=0)

    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(
            backend='nccl',  # 'distributed backend'
            init_method=
            'tcp://127.0.0.1:9999',  # distributed training init method
            world_size=1,  # number of nodes for distributed training
            rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # # Dataset
    # dataset = LoadImagesAndLabels(
    #     train_list,
    #     img_size,
    #     batch_size,
    #     augment=True,
    #     hyp=hyp,  # augmentation hyperparameters
    #     rect=opt.rect,  # rectangular training
    #     image_weights=opt.img_weights,
    #     cache_labels=True if epochs > 10 else False,
    #     cache_images=False if opt.prebias else opt.cache_images)
    # Dataloader

    train_data = DetectionDataset(
        train_list,
        './tmp/yolot',
        cache_len=1000,
        img_size=img_size,
        augments=augments,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=min([os.cpu_count(), batch_size, 16]),
        shuffle=True,
        pin_memory=True,
        collate_fn=train_data.collate_fn)
    val_data = DetectionDataset(
        val_list,
        './tmp/yolov',
        cache_len=1000,
        img_size=img_size,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=min([os.cpu_count(), batch_size, 16]),
        shuffle=True,
        pin_memory=True,
        collate_fn=val_data.collate_fn)
    # Start training
    model.nc = nc  # attach number of classes to model
    model.arc = opt.arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    nb = len(train_loader)
    maps = np.zeros(nc)  # mAP per class
    results = (
        0, 0, 0, 0, 0, 0, 0
    )  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print('Starting %s for %g epochs...' %
          ('prebias' if opt.prebias else 'training', epochs))
    while epoch < epochs:  # epoch ------------------------------------------------------------------
        model.train()
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls',
                                     'total', 'targets', 'img_size'))

        # Update image weights (optional)
        # if dataset.image_weights:
        #     w = model.class_weights.cpu().numpy() * (1 -
        #                                              maps)**2  # class weights
        #     image_weights = labels_to_image_weights(dataset.labels,
        #                                             nc=nc,
        #                                             class_weights=w)
        #     dataset.indices = random.choices(range(dataset.n),
        #                                      weights=image_weights,
        #                                      k=dataset.n)  # rand weighted idx

        pbar = tqdm(enumerate(train_loader), total=nb)  # progress bar
        for idx, (
                inputs, targets, paths, _
        ) in pbar:  # batch -------------------------------------------------------------
            total_loss = torch.zeros(4).to(device)  # mean losses
            batch_idx = idx + 1
            # TODO show_batch
            # if idx == 0:
            #     show_batch('train_batch.png', inputs, targets, classes)
            inputs = inputs.to(device)
            targets = targets.to(device)
            if multi_scale and (inputs.size(3) != img_size):
                inputs = F.interpolate(inputs,
                                       size=img_size,
                                       mode='bilinear',
                                       align_corners=False)

            # Run model
            pred = model(inputs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / 64

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Print batch results
            total_loss += loss_items
            mloss = total_loss / batch_idx  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available(
            ) else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1),
                                               '%.3gG' % mem, *mloss,
                                               len(targets), img_size)
            pbar.set_description(s)
            if batch_idx % accumulate == 0 or \
                    batch_idx == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

                # multi scale
                if multi_scale:
                    img_size = random.randrange(img_size_min,
                                                img_size_max) * 32
            # end batch ------------------------------------------------------------------------------------------------

        # Calculate mAP (always test final epoch, skip first 10 if opt.nosave)
        if not (opt.notest or (opt.nosave and epoch < 10)):
            with torch.no_grad():
                results, maps = test.test(
                    cfg,
                    data,
                    val_loader,
                    batch_size=batch_size,
                    img_size=opt.img_size,
                    model=model,
                    conf_thres=0.01 if epoch > 20 else 0.1,  # 0.1 for speed
                    save_json=epoch > 0 and 'coco.data' in data)

        # Write Tensorboard results
        if tb_writer:
            x = list(mloss) + list(results)
            titles = [
                'GIoU', 'Objectness', 'Classification', 'Train loss',
                'Precision', 'Recall', 'mAP', 'F1', 'val GIoU',
                'val Objectness', 'val Classification'
            ]
            for xi, title in zip(x, titles):
                tb_writer.add_scalar(title, xi, epoch)

        val_loss = sum(results[4:])  # total loss
        mAP = results[2]
        epoch += 1
        # Save checkpoint.
        state_dict = {
            'model': model.state_dict(),
            'mAP': mAP,
            'loss': val_loss,
            'epoch': epoch,
        }
        if opt.adam:
            state_dict['adam'] = optimizer.state_dict()
        torch.save(state_dict, 'weights/last.pt')
        if val_loss < best_loss:
            print('\nSaving best_loss.pt..')
            torch.save(state_dict, 'weights/best_loss.pt')
            best_loss = val_loss
        if mAP > best_mAP:
            print('\nSaving best_mAP.pt..')
            torch.save(state_dict, 'weights/best_mAP.pt')
            best_mAP = mAP
        if epoch % 10 == 0 and epoch > 1:
            print('\nSaving backup%d.pt..' % epoch)
            torch.save(state_dict, 'weights/backup%d.pt' % epoch)
        torch.cuda.empty_cache()
        # end epoch ----------------------------------------------------------------------------------------------------

    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs', type=int,
        default=273)  # 500200 batches at bs 16, 117263 images = 273 epochs
    parser.add_argument(
        '--batch-size', type=int,
        default=4)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate',
                        type=int,
                        default=16,
                        help='batches to accumulate before optimizing')
    parser.add_argument('--cfg',
                        type=str,
                        default='cfg/yolov3-spp.cfg',
                        help='cfg file path')
    parser.add_argument('--data',
                        type=str,
                        default='tt100k.data',
                        help='*.data file path')
    parser.add_argument('--multi-scale',
                        action='store_true',
                        help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img-size',
                        type=int,
                        default=416,
                        help='inference size (pixels)')
    parser.add_argument('--rect',
                        action='store_true',
                        help='rectangular training')
    parser.add_argument('--resume',
                        action='store_true',
                        help='resume training from last.pt')
    parser.add_argument('--transfer',
                        action='store_true',
                        help='transfer learning')
    parser.add_argument('--nosave',
                        action='store_true',
                        help='only save final checkpoint')
    parser.add_argument('--notest',
                        action='store_true',
                        help='only test final epoch')
    parser.add_argument('--evolve',
                        action='store_true',
                        help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--img-weights',
                        action='store_true',
                        help='select training images by weight')
    parser.add_argument('--cache-images',
                        action='store_true',
                        help='cache images for faster training')
    parser.add_argument(
        '--weights',
        type=str,
        default='weights/last.pt',
        help='initial weights')  # i.e. weights/darknet.53.conv.74
    parser.add_argument('--arc',
                        type=str,
                        default='default',
                        help='yolo architecture')  # defaultpw, uCE, uBCE
    parser.add_argument('--prebias',
                        action='store_true',
                        help='transfer-learn yolo biases prior to training')
    parser.add_argument(
        '--name',
        default='',
        help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device',
                        default='',
                        help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--adam',
                        action='store_true',
                        help='use adam optimizer')
    parser.add_argument('--var', type=float, help='debug variable')
    opt = parser.parse_args()
    print(opt)
    device = torch_utils.select_device(opt.device, apex=mixed_precision)

    tb_writer = None
    if not opt.evolve:  # Train normally
        try:
            # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter()
        except:
            pass

        train()  # train normally

    else:  # Evolve hyperparameters (optional)
        opt.notest = True  # only test final epoch
        opt.nosave = True  # only save final checkpoint
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' %
                      opt.bucket)  # download evolve.txt if exists

        for _ in range(1):  # generations to evolve
            if os.path.exists(
                    'evolve.txt'
            ):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                x = np.loadtxt('evolve.txt', ndmin=2)
                parent = 'weighted'  # parent selection method: 'single' or 'weighted'
                if parent == 'single' or len(x) == 1:
                    x = x[fitness(x).argmax()]
                elif parent == 'weighted':  # weighted combination
                    n = min(10, x.shape[0])  # number to merge
                    x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                    w = fitness(x) - fitness(x).min()  # weights
                    x = (x[:n] *
                         w.reshape(n, 1)).sum(0) / w.sum()  # new parent
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = x[i + 7]

                # Mutate
                np.random.seed(int(time.time()))
                s = [
                    .2, .2, .2, .2, .2, .2, .2, .0, .02, .2, .2, .2, .2, .2,
                    .2, .2, .2, .2
                ]  # sigmas
                for i, k in enumerate(hyp.keys()):
                    x = (np.random.randn(1) * s[i] +
                         1)**2.0  # plt.hist(x.ravel(), 300)
                    hyp[k] *= float(x)  # vary by sigmas

            # Clip to limits
            keys = [
                'lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v',
                'translate', 'scale', 'fl_gamma'
            ]
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001),
                      (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            results = train()

            # Write mutation results
            print_mutation(hyp, results, opt.bucket)

            # Plot results
            # plot_evolution_results(hyp)
