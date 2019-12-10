from sklearn.cluster import KMeans
import os
import numpy as np
from tqdm import tqdm
import argparse


def kmeans_anchor(label_dir, n_clusters=9, img_size=(320, 180)):
    names = os.listdir(label_dir)
    names = [name for name in names if os.path.splitext(name)[-1] == '.txt']
    wh_list = []
    for name in tqdm(names):
        with open(os.path.join(label_dir, name), 'r') as f:
            lines = f.read().split('\n')
            lines = [l.split(' ') for l in lines]
        for line in lines:
            if len(line) != 5:
                continue
            wh_list.append([float(i) for i in line[3:]])
    wh_list = np.float32(wh_list)
    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(wh_list)
    anchors = np.float32(estimator.cluster_centers_)
    anchors[:, 0] *= img_size[0]
    anchors[:, 1] *= img_size[1]
    area = anchors[:, 0] * anchors[:, 1]
    
    output = 'anchors: '
    for i in range(n_clusters):
        index = np.argmax(area)
        area[index] = 0
        output += '[%d,%d], ' % (int(anchors[index][0]), int(anchors[index][1]))
    print(output[:-2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--labels', type=str)
    parser.add_argument('-n', type=int, default=9)
    parser.add_argument('-s', '--img-size', type=str, default='320')
    opt = parser.parse_args()
    print(opt)
    size = opt.img_size.split(',')
    if len(size) == 1:
        size *= 2
    size = [int(s) for s in size]
    kmeans_anchor(opt.labels, opt.n, size)
