# -*- coding: UTF-8 -*-

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

def do_pca(X, n_components, whiten=True):
    print("do_pca...")
    pca = PCA(n_components=n_components, whiten=whiten)
    print("   PCA train...")
    pca.fit(X)
    print("   PCA transform...")
    pca_data = pca.transform(X)
    # pca_data = pca.fit_transform(X)
    return pca_data


def do_som(data, x, y, input_len, learning_rate=0.5):
    '''自组织映射 (Self-Organizing Map, SOM)'''
    from minisom import MiniSom
    print("do_som...")
    som = MiniSom(x=x, y=y, input_len=input_len, learning_rate=learning_rate)
    som.random_weights_init(data) # 随机初始化权重
    print("   som train...")
    som.train_random(data, num_iteration=100) # 训练

    # low_dim_data = som.distance_map() # 降维数据
    print("   som winner...")
    bmus = np.array([som.winner(x) for x in data])

    return bmus

def do_tSNE(X, n_components):
    print("do_TSNE...")
    tsne = TSNE(n_components=n_components)
    # tsne.fit(X)
    X_embedded = tsne.fit_transform(X)
    return X_embedded
