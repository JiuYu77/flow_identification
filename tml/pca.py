# -*- coding: UTF-8 -*-

from sklearn.decomposition import PCA

def do_pca(X, n_components, whiten=True):
    print("do_pca...")
    pca = PCA(n_components=n_components, whiten=whiten)
    print("   PCA train...")
    pca.fit(X)
    print("   PCA transform...")
    pca_data = pca.transform(X)
    # pca_data = pca.fit_transform(X)
    return pca_data
