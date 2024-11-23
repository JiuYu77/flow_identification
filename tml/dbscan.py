# -*- coding: UTF-8 -*-
"""DBSCAN 密度聚类"""

from sklearn.cluster import DBSCAN

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from utils import FlowDataset


dataset = FlowDataset("../dataset/v4/Pressure/v4_Pressure_Simple/4/val", 4096, 2048, cls=-1)
all = dataset.allSample
data = np.array(all)


# ######################
from minisom import MiniSom
som = MiniSom(x=len(all), y=32, input_len=4096, learning_rate=0.5)

def std(x):
    mean = np.mean(x)
    std = np.nanstd(x)
    std = max(std, 0.001)
    x = (x - mean) / std
    return x
for i,v in enumerate(data):
    x = std(v)
    data[i] = x

som.random_weights_init(data) # 随机初始化权重
print("som train...")
som.train_random(data, num_iteration=2) # 训练

# low_dim_data = som.distance_map() # 降维数据
# low_dim_data = som.quantization(data)
print("som winner...")
low_dim_data = bmus = np.array([som.winner(x) for x in data])
print(bmus.shape)
# ######################

print("DBSCAN...")
# dbscan = DBSCAN(eps=12, min_samples=5)
# dbscan = DBSCAN(eps=7, min_samples=5)
dbscan = DBSCAN(eps=7, min_samples=10)
# X = data
X = low_dim_data
dbscan.fit(X)

unique_elements = np.unique(dbscan.labels_)
print("unique_elements:", unique_elements)
print("类别数量:", len(unique_elements))

with open("tml/aaa.txt", 'w') as f:
    i = 0
    for v in dbscan.labels_:
        ss = str(v)+' '
        i += 1
        if i >= 40:
            i = 0
            ss += "\n"
        f.write(ss)
print(dbscan.labels_);exit()


# 可视化散点图
fig = plt.figure()
ax = fig.add_subplot(111)
ax = fig.add_subplot(111, projection='3d')

# 设置标签
ax.set_xlabel('Feature 0')
ax.set_ylabel('Feature 1')
# ax.set_zlabel('Feature 2')

# 绘制散点图
# ax.scatter(data[:, 0], data[:, 1], data[:,2])
ax.scatter(data[:, 0], data[:, 1], data[:,2],c=dbscan.labels_)
print(dbscan.labels_)
plt.show()
plt.savefig("tml/aaa")

