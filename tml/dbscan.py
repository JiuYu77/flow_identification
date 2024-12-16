# -*- coding: UTF-8 -*-
"""DBSCAN 密度聚类"""

from sklearn.cluster import DBSCAN
import numpy as np

def do_dbscan(X, eps, min_samples):
    print("do_dbscan...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)

    unique_elements = np.unique(dbscan.labels_)
    print("   unique_elements:", unique_elements)
    print("   类别数量:", len(unique_elements))
    return dbscan.labels_
