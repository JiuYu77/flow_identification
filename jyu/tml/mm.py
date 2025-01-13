import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.stats import mode

import sys
sys.path.append('.')

from jyu.utils import FlowDataset, ph
from dimensionality_reduction import do_pca, do_som, do_tSNE
from cluster import *
from utils import *


datasetPath = "../dataset/v4/Pressure/v4_Pressure_Simple/4/train"
datasetPath = "/home/jyu/apro/flow/dataset/flow/v4/Pressure/v4/train"
# datasetPath = "/home/jyu/apro/flow/dataset/flow/v4/Pressure/v4/val"

dataset = FlowDataset(datasetPath, 4096, 2048, "zScore_std")
dataset.do_shuffle()
dataset.allSample = np.array(dataset.allSample)
# dataset.do_transform()
all = dataset.allSample
data = all

# ######################

# pca_data = do_pca(data, 4, True)
# print('PCA降维后数据:')
# print(pca_data.shape)
# print(pca_data)
# low_dim_data = pca_data

low_dim_data = do_tSNE(data, 3)

# 使用多个聚类算法
X = low_dim_data
labels_kmeans = do_k_means(X, 7, 'auto')
labels_agglomerative = do_Agglomerative(X, 7)
labels_mini = do_MiniBatchKMeans(X, 7, 'auto')
labels_birch = do_birch(X, 7)
labels_spectral = do_SpectralClustering(X, 7)
labels_gmm = do_GaussianMixture(X, 7)

# 将所有聚类结果汇总
all_labels = np.vstack((labels_kmeans, labels_agglomerative, labels_mini, labels_birch, labels_spectral, labels_gmm))

# 聚类投票：对每个数据点的标签取众数
final_labels = mode(all_labels, axis=0, keepdims=True).mode[0]

# 打印或分析最终聚类结果
print("Final cluster labels:", final_labels)
PseudoLabel = final_labels

path = "tmp/tml/pca-mm/"
ph.checkAndInitPath(path)

save_label(path + "pre.txt", PseudoLabel)
save_label(path + "true.txt", dataset.allLabel)

print("dbscan.labels_: ", PseudoLabel)

# 可视化散点图
draw(X, PseudoLabel, path+'pre')
draw(X, dataset.allLabel, path + "true")

