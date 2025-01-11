# -*- coding: UTF-8 -*-
"""K-Means 聚类"""

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('.')

from pca import do_pca
from som import do_som
from minibatch_k_means import do_MiniBatchKMeans
from jyu.utils import FlowDataset, ph

def draw(data, c, path):
    # 可视化散点图
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 设置标签
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')

    # 绘制散点图
    # ax.scatter(data[:, 0], data[:, 1])
    ax.scatter(data[:, 0], data[:, 1], c=c)
    plt.savefig(path)
    plt.show()
    plt.close(fig)

def draw3d(data, c, path):
    # 可视化散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置标签
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.set_zlabel('Feature 2')

    # 绘制散点图
    # ax.scatter(data[:, 0], data[:, 1], data[:,2])
    ax.scatter(data[:, 0], data[:, 1], data[:,2], c=c)

    plt.savefig(path)
    plt.show()
    plt.close(fig)

def std(x):
    mean = np.mean(x)
    std = np.nanstd(x)
    std = max(std, 0.001)
    x = (x - mean) / std
    return x

def do_shuffle(data, label):
    indices = np.random.permutation(len(data))
    data_shuffled = data[indices]
    if type(label) is list:
        label = np.array(label)
    label_shuffled = label[indices]
    return data_shuffled, label_shuffled

datasetPath = "../dataset/v4/Pressure/v4_Pressure_Simple/4/train"
datasetPath = "/home/jyu/apro/flow/dataset/flow/v4/Pressure/v4/train"
datasetPath = "/home/jyu/apro/flow/dataset/flow/v4/Pressure/v4/val"

# dataset = FlowDataset(datasetPath, 4096, 2048, cls=-1)
# dataset = FlowDataset(datasetPath, 4096, 2048)
# dataset = FlowDataset(datasetPath, 4096, 2048, "dwt_zScore")
dataset = FlowDataset(datasetPath, 4096, 4096, "zScore_std")
dataset.allSample = np.array(dataset.allSample)
dataset.do_transform()

# all = dataset.allSample
all, label = do_shuffle(dataset.allSample, dataset.allLabel)
data = all

# ######################

pca_data = do_pca(data, 32, True)
print('PCA降维后数据:')
print(pca_data.shape)
print(pca_data)

low_dim_data = pca_data

# for i,v in enumerate(pca_data):
#     x = std(v)
#     pca_data[i] = x

# som_dim = int(math.sqrt(pca_data.shape[0]) + 0.5)
# low_dim_data = do_som(pca_data, som_dim, som_dim, pca_data.shape[1], 0.5)

# from sklearn.manifold import TSNE
# print("TSNE...")
# tsne = TSNE(n_components=2)
# # tsne.fit(data)
# # low_dim_data = tsne.fit_transform(data)
# tsne.fit(pca_data)
# low_dim_data = tsne.fit_transform(pca_data)

# ##################################################

X = low_dim_data
# PseudoLabel = do_dbscan(X, eps=0.5, min_samples=5)
# PseudoLabel = do_dbscan(X, eps=1, min_samples=10)
# PseudoLabel = do_dbscan(X, eps=0.3, min_samples=100)
# PseudoLabel = do_dbscan(X, eps=0.5, min_samples=10)

PseudoLabel = do_MiniBatchKMeans(X, 7, 'auto')

path = "tmp/tml/pca-som-MBkmeans/"
ph.checkAndInitPath(path)

def save_label(p, labels):
    with open(p, 'w') as f:
        i = 0
        for v in labels:
            ss = str(v)+' '
            i += 1
            if i >= 40:
                i = 0
                ss += "\n"
            f.write(ss)

save_label(path + "pre.txt", PseudoLabel)
save_label(path + "true.txt", dataset.allLabel)
save_label(path + "true-shuffle.txt", label)

print("dbscan.labels_: ", PseudoLabel)

# 可视化散点图
draw(X, PseudoLabel, path+'pre')
draw(X, dataset.allLabel, path + "true")
draw(X, label, path + "true-shuffle")

