# -*- coding: UTF-8 -*-
"""DBSCAN 密度聚类"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from pca import do_pca
from som import do_som
from dbscan import do_dbscan
from utils import FlowDataset, ph

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
    plt.show()
    plt.savefig(path)
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

    plt.show()
    plt.savefig(path)
    plt.close(fig)

def std(x):
    mean = np.mean(x)
    std = np.nanstd(x)
    std = max(std, 0.001)
    x = (x - mean) / std
    return x

# dataset = FlowDataset("../dataset/v4/Pressure/v4_Pressure_Simple/4/val", 4096, 2048, cls=-1)
# dataset = FlowDataset("../dataset/v4/Pressure/v4_Pressure_Simple/4/val", 4096, 2048)
dataset = FlowDataset("../dataset/v4/Pressure/v4_Pressure_Simple/4/train", 4096, 2048, "dwt_zScore")
dataset.allSample = np.array(dataset.allSample)
dataset.do_transform()

all = dataset.allSample
data = all

# ######################

pca_data = do_pca(data, 128, True)
print('PCA降维后数据:')
print(pca_data.shape)
print(pca_data)

# for i,v in enumerate(pca_data):
#     x = std(v)
#     pca_data[i] = x

# som_dim = int(math.sqrt(pca_data.shape[0]) + 0.5)
# low_dim_data = do_som(pca_data, som_dim, som_dim, pca_data.shape[1], 0.5)

from sklearn.manifold import TSNE
print("TSNE...")
tsne = TSNE(n_components=2)
# tsne.fit(data)
# low_dim_data = tsne.fit_transform(data)
tsne.fit(pca_data)
low_dim_data = tsne.fit_transform(pca_data)

# ##################################################

X = low_dim_data
# PseudoLabel = do_dbscan(X, eps=0.5, min_samples=5)
PseudoLabel = do_dbscan(X, eps=1, min_samples=10)
# PseudoLabel = do_dbscan(X, eps=0.3, min_samples=100)
# PseudoLabel = do_dbscan(X, eps=0.5, min_samples=10)


path = "tmp/tml/pca-som-dbscan/"
ph.checkAndInitPath(path)

with open(path + "aaa.txt", 'w') as f:
    i = 0
    for v in PseudoLabel:
        ss = str(v)+' '
        i += 1
        if i >= 40:
            i = 0
            ss += "\n"
        f.write(ss)
with open(path + "true.txt", 'w') as f:
    i = 0
    for v in dataset.allLabel:
        ss = str(v)+' '
        i += 1
        if i >= 40:
            i = 0
            ss += "\n"
        f.write(ss)
print("dbscan.labels_: ", PseudoLabel)

# 可视化散点图
draw(X, PseudoLabel, path+'aaa')
draw(X, dataset.allLabel, path + "true")

