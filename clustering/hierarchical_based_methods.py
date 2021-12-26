from sklearn.cluster import AgglomerativeClustering

def agglomerative_clustering(data):
    cluster = AgglomerativeClustering(n_clusters=5)
    return cluster.fit_predict(data)