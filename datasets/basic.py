import os
import torch
from random import randint
from threading import Thread
from queue import Queue


class BasicDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 cache_dir=None,
                 cache_len=3000,
                 img_size=224,
                 augments={}):
        super(BasicDataset, self).__init__()
        self.path = path
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache = True
        else:
            self.cache = False
        self.img_size = img_size
        self.augments = augments
        self.data = []
        self.classes = []
        self.build_data()
        if self.cache:
            self.counts = [0 for i in self.data]
            self.cache_path = [
                os.path.join(cache_dir, str(i)) for i in range(len(self.data))
            ]
            self.cache_len = cache_len
            self.cache_memory = [None for i in range(cache_len)]
            self.cache_worker_queue = Queue(0)
            t = Thread(target=self.worker)
            t.setDaemon(True)
            t.start()

    def worker(self):
        while True:
            idx = self.cache_worker_queue.get()
            self.refresh_cache(idx)
            self.cache_worker_queue.task_done()

    def refresh_cache(self, idx):
        item = self.get_item(idx)
        if idx < self.cache_len:
            self.cache_memory[idx] = item
        torch.save(item, self.cache_path[idx])
        self.counts[idx] = 0
        return item

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.cache:
            return self.get_item(idx)
        self.counts[idx] += 1
        item = None
        if idx < self.cache_len:
            if self.cache_memory[idx] is not None:
                item = self.cache_memory[idx]
        if item is None:
            try:
                item = torch.load(self.cache_path[idx])
                assert item[0].size(0) == self.img_size
            except:
                item = self.refresh_cache(idx)
        if self.counts[idx] > randint(1, 5):
            self.cache_worker_queue.put(idx)
        return item

    def build_data(self):
        pass

    def get_item(self, idx):
        return None
