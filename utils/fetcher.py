import torch
from threading import Thread, Lock
from queue import Queue, Empty, Full
from time import sleep, time
from . import device


# Referencehttps://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
class Fetcher:
    def __init__(self, loader, max_len=5):
        self.idx = 0
        self.loader = loader
        self.loader_iter = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return self

    def preload(self):
        try:
            self.batch = next(self.loader_iter)
        except StopIteration:
            self.loader_iter = iter(self.loader)
            self.batch = next(self.loader_iter)
        if device == 'cuda':
            with torch.cuda.stream(self.stream):
                self.batch = [
                    b.cuda(non_blocking=True)
                    if isinstance(b, torch.Tensor) else b for b in self.batch
                ]

    def __next__(self):
        if device == 'cuda':
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch
