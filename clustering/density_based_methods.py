from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import math
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import random
import numpy as np
def dbscan_clustering(data):
    dbscan = DBSCAN(eps=2.0)
    label = dbscan.fit_predict(data)
    return label

def optics_clustering(data):
    optics = OPTICS()
    return optics.fit_predict(data)


def dist(a, b):
    len_a = len(a)
    ans = 0
    for i in range(len_a):
        ans = ans + (a[i] - b[i]) * (a[i] - b[i])
    return math.sqrt(ans)

def getCore(dataset , eps , min_samples):
    core = {}
    KdTree = KDTree(dataset)
    len_d = len(dataset)
    for i in range(len_d):
        temp = np.sum((dataset - dataset[i]) ** 2, axis=1) ** 0.5
        neibor = np.argwhere(temp <= eps).flatten().tolist()
        #neibor = KdTree.query_ball_point(dataset[i],eps)
        if len(neibor) > min_samples :
            core[i] = neibor
    return core

def dbscan(dataset,eps = 1.8,min_samples = 8):
    len_d = len(dataset)
    '''
    dis_k = []

    for idx1 in range(len_d):
        dis = []
        for idx2 in range(len_d):
            dis.append(dist(dataset[idx1],dataset[idx2]))
        dis = sorted(dis)
        dis_k.append(dis[4])
    dis_k = sorted(dis_k)
    x1 = [i for i in range(len_d)]
    plt.plot(x1, dis_k, label='Frist line', linewidth=3, color='r', marker='o',markerfacecolor='blue', markersize=12)
    plt.xlabel('Plot Number')
    plt.ylabel('distance')
    plt.title('Eps Selection')
    plt.legend()
    plt.show()
    '''
    unvisited = [0 for i in range(len_d)]
    core_neiborhood = getCore(dataset , eps ,min_samples)
    #print(core_neiborhood)
    cluster = [-1 for i in range(len_d)] #记录结果
    cluster_id = 0
    for core_id,neibor in core_neiborhood.items(): #遍历核心点
        if unvisited[core_id] == 1: #如果已经访问过
            continue
        new_cluster = [core_id]
        core_queue = [core_id]
        unvisited[core_id] = 1
        while len(core_queue): #对核心点进行bfs
            x = core_queue.pop()
            neibor = core_neiborhood[x]
            for n_id in neibor: #核心点邻域
                if unvisited[n_id] == 0: #如果发现没访问过
                    unvisited[n_id] = 1
                    new_cluster.append(n_id)
                    if n_id in core_neiborhood : #如果发现邻域内核心点
                        core_queue.append(n_id)
        for kk in new_cluster:
            cluster[kk] = cluster_id
        cluster_id = cluster_id + 1

    #print(cluster)
    print(cluster_id)
    return cluster


