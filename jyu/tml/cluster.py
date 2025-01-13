# -*- coding: UTF-8 -*-

from sklearn.cluster import (
    DBSCAN,
    KMeans, MiniBatchKMeans,
    AgglomerativeClustering,
    Birch,
    SpectralClustering
)
from sklearn.mixture import GaussianMixture
import numpy as np

def do_dbscan(X, eps, min_samples):
    print("do_dbscan...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)

    unique_elements = np.unique(dbscan.labels_)
    print("   unique_elements:", unique_elements)
    print("   类别数量:", len(unique_elements))
    return dbscan.labels_

def do_k_means(X, n_clusters, n_init='warn', max_iter=1000, random_state=None):
    print("do_kmeans...")
    k_means = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, random_state=random_state)
    Kmeans = k_means.fit(X)
    pre = k_means.predict(X)
    return pre

def do_MiniBatchKMeans(X, n_clusters, n_init, batchSize=1024, max_iter=1000):
    print("do_MiniBatchKMeans...")
    mb_kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=n_init, batch_size=batchSize, max_iter=max_iter)
    # Kmeans = mb_kmeans.fit(X)
    # pre = mb_kmeans.predict(X)
    pre = mb_kmeans.fit_predict(X)
    return pre

def do_Agglomerative(X, n_clusters):
    print("do_AgglomerativeClustering...")
    agg = AgglomerativeClustering(n_clusters=n_clusters,linkage='ward') # 最近的距离为标准
    agg.fit(X)
    return agg.labels_

def do_birch(X, n_clusters):
    print("do_Birch...")
    birch = Birch(n_clusters=n_clusters)
    y_pred = birch.fit_predict(X)
    return y_pred

def do_SpectralClustering(X, n_clusters, affinity='nearest_neighbors', random_state=42):
    print("do_SpectralClustering...")
    spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=random_state)
    # spectral = SpectralClustering(n_clusters=n_clusters)
    labels_spectral = spectral.fit_predict(X)
    return labels_spectral

def do_GaussianMixture(X, n_components, random_state=42):
    print("do_GaussianMixture...")
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    # gmm = GaussianMixture(n_components=n_components)
    labels_gmm = gmm.fit_predict(X)
    return labels_gmm
