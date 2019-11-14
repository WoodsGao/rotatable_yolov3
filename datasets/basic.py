import torch
from torch.multiprocessing import Manager, Value
import random


class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, path, cache_size=0, img_size=224, augments={}):
        super(BasicDataset, self).__init__()
        self.path = path
        self.img_size = img_size
        self.augments = augments
        self.data = []
        self.classes = []
        self.build_data()
        self.max_cache_size = cache_size * 1e6
        self.manager = Manager()
        self.cache_list = self.manager.list([None for i in range(len(self.data))])
        self.cache_size = Value('l', self.get_cache_size(self.cache_list))

    def get_cache_size(self, data):
        size = 0
        for d in data:
            if d is None:
                continue
            if isinstance(d, tuple) or isinstance(d, list):
                size += self.get_cache_size(d)
            elif isinstance(d, torch.Tensor):
                size += d.storage().__sizeof__()
            else:
                size += d.__sizeof__()
        return size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.cache_list[idx] is not None:
            if random.random() < 0.5:
                item = self.cache_list[idx]
            else:
                item = self.get_item(idx)
                self.cache_list[idx] = item
        else:
            item = self.get_item(idx)
            if self.cache_size.value < self.max_cache_size:
                self.cache_list[idx] = item
                self.cache_size.value = self.get_cache_size(self.cache_list)
        return item

    def build_data(self):
        pass

    def get_item(self, idx):
        return None
