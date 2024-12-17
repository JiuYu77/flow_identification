from sklearn.cluster import KMeans

def do_k_means(X, n_clusters, n_init, max_iter=1000):
    print("do_means...")
    k_means = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    Kmeans = k_means.fit(X)
    pre = k_means.predict(X)
    return pre
