import os
import torch
import torch.optim as optim
from tqdm import tqdm
from time import time
from . import device
amp = None
try:
    from apex import amp
except ImportError:
    pass
if device == 'cuda':
    torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self,
                 model,
                 fetcher,
                 loss_fn,
                 weights='',
                 accumulate=1,
                 adam=False,
                 lr=0,
                 distributed=False,
                 mixed_precision=False):
        self.accumulate_count = 0
        self.metrics = 0
        self.epoch = 0
        self.accumulate = accumulate
        model = model.to(device)
        self.fetcher = fetcher
        self.loss_fn = loss_fn
        if adam:
            optimizer = optim.AdamW(model.parameters(),
                                    lr=lr if lr > 0 else 1e-4,
                                    weight_decay=1e-5)
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=lr if lr > 0 else 1e-3,
                                  momentum=0.9,
                                  weight_decay=1e-5,
                                  nesterov=True)
        self.adam = adam
        if amp is None:
            self.mixed_precision = False
        else:
            self.mixed_precision = mixed_precision
        if self.mixed_precision:
            model, optimizer = amp.initialize(model,
                                              optimizer,
                                              opt_level='O1',
                                              verbosity=0)
        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True)
        self.model = model
        self.optimizer = optimizer
        if weights:
            self.load(weights)
            if lr > 0:
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr
        self.optimizer.zero_grad()
        if distributed:
            self.model.require_backward_grad_sync = False
        self.distributed = distributed

    def load(self, weights):
        state_dict = torch.load(weights, map_location=device)
        if self.adam:
            if 'adam' in state_dict:
                self.optimizer.load_state_dict(state_dict['adam'])
        else:
            if 'sgd' in state_dict:
                self.optimizer.load_state_dict(state_dict['sgd'])
        if 'm' in state_dict:
            self.metrics = state_dict['m']
        if 'e' in state_dict:
            self.epoch = state_dict['e']
        self.model.load_state_dict(state_dict['model'], strict=False)

    def save(self, save_path_list):
        if len(save_path_list) == 0:
            return False
        state_dict = {
            'model': self.model.state_dict(),
            'm': self.metrics,
            'e': self.epoch
        }
        if self.adam:
            state_dict['adam'] = self.optimizer.state_dict()
        else:
            state_dict['sgd'] = self.optimizer.state_dict()
        for save_path in save_path_list:
            torch.save(state_dict, save_path)

    def run_epoch(self):
        print('Epoch: %d' % self.epoch)
        self.model.train()
        total_loss = 0
        c = 0
        t = [0, 0, 0, 0]
        pbar = tqdm(enumerate(self.fetcher), total=len(self.fetcher))
        t0 = time()
        for idx, (inputs, targets) in pbar:
            t1 = time()
            if inputs.size(0) < 2:
                continue
            c += 1
            self.accumulate_count += 1
            batch_idx = idx + 1
            if self.accumulate_count % self.accumulate == 0 and self.distributed:
                self.model.require_backward_grad_sync = True
            outputs = self.model(inputs)
            t2 = time()
            loss = self.loss_fn(outputs, targets)
            total_loss += loss.item()
            loss /= self.accumulate
            t3 = time()
            # Compute gradient
            if self.mixed_precision:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            t4 = time()
            t[0] += t1 - t0
            t[1] += t2 - t1
            t[2] += t3 - t2
            t[3] += t4 - t3
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available(
            ) else 0  # (GB)
            pbar.set_description(
                'mem: %8g, loss: %8g, scale: %8g, %6g %6g %6g %6g' %
                (mem, total_loss / batch_idx, inputs.size(2), t[0] / c,
                 t[1] / c, t[2] / c, t[3] / c))
            if self.accumulate_count % self.accumulate == 0:
                self.accumulate_count = 0
                # print(self.model.module.backbone.block1[1].blocks[0].block[1].conv.weight)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.distributed:
                    self.model.require_backward_grad_sync = False
            t0 = time()
        torch.cuda.empty_cache()
        self.epoch += 1
        return total_loss / len(self.fetcher)