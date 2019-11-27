import torch
from threading import Thread, Lock
from queue import Queue, Empty, Full
from time import sleep, time
from . import device


# Referencehttps://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
class Fetcher:
    def __init__(self, loader, post_fetch_fn=None):
        self.idx = 0
        self.loader = loader
        self.loader_iter = iter(loader)
        self.post_fetch_fn = post_fetch_fn
        if device == 'cuda':
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
            self.batch = None
            self.loader_iter = iter(self.loader)
            return None
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
        if batch is None:
            raise StopIteration
        if self.post_fetch_fn is not None:
            batch = self.post_fetch_fn(batch)
        return batch
