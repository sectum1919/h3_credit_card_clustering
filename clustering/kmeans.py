from sklearn.cluster import KMeans

def kmeans_clustering(data):
    k_means = KMeans()
    k_means.fit(data)
    return k_means.predict(data)