import torch
import torch.utils.data
from tqdm import tqdm
import os
import lmdb
import pickle
import base64
import time
from threading import Thread
from concurrent.futures import ThreadPoolExecutor


class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, path, img_size=224, augments={}, skip_init=False):
        super(BasicDataset, self).__init__()
        os.makedirs('tmp', exist_ok=True)
        self.path = path
        self.img_size = img_size
        self.augments = augments
        self.data = []
        self.classes = []
        self.build_data()
        self.data.sort()
        self.checks = [[d, self.img_size] for d in self.data]
        self.cache_list = []
        self.db_name = 'tmp/' + base64.b64encode(
            os.path.abspath(path).encode('utf-8')).decode('utf-8')
        if not skip_init:
            self.init_db()
            if len(augments) > 0:
                p = Thread(target=self.worker, daemon=True)
                p.start()

    def init_db(self):
        # init
        min_key = 0
        max_key = len(self.data)
        dump_keys = []
        db = lmdb.open(self.db_name, map_size=1e12, writemap=True)
        txn = db.begin(write=True)
        print('check db')
        for key, value in tqdm(txn.cursor()):
            ki = int.from_bytes(key, byteorder='little')
            if min_key <= ki and ki < max_key:
                try:
                    check = pickle.loads(value)[1]
                    if check == self.checks[ki]:
                        continue
                except Exception as e:
                    print(e)
            dump_keys.append(key)
        for key in dump_keys:
            txn.delete(key)
        missed_keys = [
            i for i in range(len(self.data))
            if txn.get(i.to_bytes(10, 'little')) is None
        ]
        batch_size = 64
        with ThreadPoolExecutor() as pool:
            for ki in tqdm(range(0, len(missed_keys), batch_size)):
                items = list(
                    pool.map(self.get_item, missed_keys[ki:ki + batch_size]))
                items = list(
                    pool.map(
                        pickle.dumps,
                        [[item, self.checks[key]]
                         for item, key in zip(items, missed_keys[ki:ki +
                                                                 batch_size])
                         ]))
                for item, key in zip(items, missed_keys[ki:ki + batch_size]):
                    txn.put(key.to_bytes(10, 'little'), value=item)
                # break
        txn.commit()
        db.close()

    def worker(self):
        while True:
            for idx, data in enumerate(self.data):
                item = self.get_item(idx)
                item = pickle.dumps([item, self.checks[idx]])
                with lmdb.open(self.db_name, map_size=1e12) as db:
                    with db.begin(write=True) as txn:
                        txn.put(idx.to_bytes(10, 'little'), value=item)
                time.sleep(0.1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with lmdb.open(self.db_name, map_size=1e12) as db:
            with db.begin() as txn:
                item = pickle.loads(txn.get(idx.to_bytes(10, 'little')))[0]
        return item

    def build_data(self):
        pass

    def get_item(self, idx):
        return None
