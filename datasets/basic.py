import torch
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, path, cache_size=0, img_size=224, augments={}):
        super(BasicDataset, self).__init__()
        self.path = path
        self.img_size = img_size
        self.augments = augments
        self.data = []
        self.classes = []
        self.build_data()
        self.cache_list = []
        pool = ThreadPoolExecutor()
        step = 64
        if cache_size > 0:
            print('preloading')
            pbar = tqdm(range(0, len(self.data), step))
            for idx in pbar:
                self.cache_list += list(
                    pool.map(self.get_item, range(idx, idx + step)))
                size = self.get_cache_size(self.cache_list) / 1e6
                pbar.set_description('%10g/%10g' % (size, cache_size))
                if size > cache_size:
                    print('preloader')
                    break

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
        if idx < len(self.cache_list) and random.random() < 0.5:
            item = self.cache_list[idx]
        else:
            item = self.get_item(idx)
        return item

    def build_data(self):
        pass

    def get_item(self, idx):
        return None
