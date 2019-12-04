import os
import torch
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
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
        self.model = model
        self.optimizer = optimizer
        if weights:
            self.load(weights)
            if lr > 0:
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr
        if self.mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model,
                                                        self.optimizer,
                                                        opt_level='O1',
                                                        verbosity=0)
        if dist.is_initialized():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, find_unused_parameters=True)
        self.optimizer.zero_grad()
        if dist.is_initialized():
            self.model.require_backward_grad_sync = False

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
            'model':
            self.model.module.state_dict()
            if dist.is_initialized() else self.model.state_dict(),
            'm':
            self.metrics,
            'e':
            self.epoch
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
        pbar = tqdm(enumerate(self.fetcher), total=len(self.fetcher))
        for idx, (inputs, targets) in pbar:
            if inputs.size(0) < 2:
                continue
            self.accumulate_count += 1
            batch_idx = idx + 1
            if self.accumulate_count % self.accumulate == 0 and dist.is_initialized():
                self.model.require_backward_grad_sync = True
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets, self.model)
            total_loss += loss.item()
            loss /= self.accumulate
            # Compute gradient
            if self.mixed_precision:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available(
            ) else 0  # (GB)
            pbar.set_description('mem: %8g, loss: %8g, scale: %8g' %
                                 (mem, total_loss / batch_idx, inputs.size(2)))
            if self.accumulate_count % self.accumulate == 0:
                self.accumulate_count = 0
                # print(self.model.module.backbone.block1[1].blocks[0].block[1].conv.weight)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if dist.is_initialized():
                    self.model.require_backward_grad_sync = False
        torch.cuda.empty_cache()
        self.epoch += 1
        return total_loss / len(self.fetcher)
