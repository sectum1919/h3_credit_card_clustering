import sklearn
import sklearn.metrics
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan

# from jqmcvi import base


def dunn(data, label):
  
    max_cluster = max(label)
    
    cluster_list = []
    for i in range(max_cluster+1):
        cluster_list.append([])
    for index, row in enumerate(data):
        cluster_list[label[index]].append(row)
    
    print(base.dunn(cluster_list))

 
# https://github.com/romusters/hopkins/blob/master/main.py
def hopkins(X):
    d = X.shape[1]# columns
    n = len(X) # rows
    m = int(0.1 * n) # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
    rand_X = sample(range(0, n, 1), m)
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        H = 0
    return H

"""metric=
    "cityblock": manhattan_distances,
    "cosine": cosine_distances,
    "euclidean": euclidean_distances,
    "haversine": haversine_distances,
    "l2": euclidean_distances,
    "l1": manhattan_distances,
    "manhattan": manhattan_distances,
    "precomputed": None,  # HACK: precomputed is always allowed, never called
    "nan_euclidean": nan_euclidean_distances,
"""

def unsupervised_evaluate(data, label):
    # https://www.biaodianfu.com/cluster-score.html

    # Silhouette Coefficient, bigger is better
    score = sklearn.metrics.silhouette_score(data, label, metric='cosine')
    print(f"Silhouette Coefficient : {score}")

    # Calinski-Harabasz Index, bigger is better
    score = sklearn.metrics.calinski_harabasz_score(data, label)
    print(f"Calinski-Harabasz Index: {score}")

    # Davies-Bouldin Index(DB/DBI), smaller is better
    score = sklearn.metrics.davies_bouldin_score (data, label)
    print(f"Davies-Bouldin Index   : {score}")

    # Dunn Validity Index(DVI), bigger is better
    # score = dunn(data, label)
    # print(f"Dunn Validity Index   : {score}")

    # https://blog.csdn.net/weixin_39671140/article/details/114690991
    # Hopkins Statistic, bigger is better
    score = hopkins(pd.DataFrame(data))
    print(f"Hopkins Statistic      : {score}")