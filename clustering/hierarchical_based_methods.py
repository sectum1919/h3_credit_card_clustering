from sklearn.cluster import AgglomerativeClustering
import math

def agglomerative_clustering(data):
    cluster = AgglomerativeClustering(n_clusters=5)
    return cluster.fit_predict(data)

def dist(a, b):
    len_a = len(a)
    ans = 0
    for i in range(len_a):
        ans = ans + (a[0] - b[0]) * (a[0] - b[0])
    return math.sqrt(ans)

def find_Min(M):
    min = 1000
    x = 0; y = 0
    for i in range(len(M)):
        for j in range(len(M[i])):
            if i != j and M[i][j] < min:
                min = M[i][j];x = i; y = j
    return (x, y, min)

#dist_min
def dist_min(Ci, Cj):
    return min(dist(i, j) for i in Ci for j in Cj)

def dist_max(Ci, Cj):
    return max(dist(i, j) for i in Ci for j in Cj)

def dist_avg(Ci, Cj):
    return sum(dist(i, j) for i in Ci for j in Cj)/(len(Ci)*len(Cj))

def AGNES(dataset, k=5):
    #初始化C和M
    dist_tmp = dist_min
    C = [];M = []
    for idx,i in enumerate(dataset):
        Ci = []
        Ci.append(idx)
        C.append(Ci)
    for i in C:
        Mi = []
        for j in C:
            Mi.append(dist_tmp(dataset[i], dataset[j]))
        M.append(Mi)
    q = len(dataset)
    len_d = q
    #合并更新
    while q > k:
        x, y, min = find_Min(M)
        C[x].extend(C[y])
        C = C[:y] + C[y+1:]
        #C.remove(C[y])
        M = []
        for i in C:
            Mi = []
            for j in C:
                Mi.append(dist_tmp(dataset[i], dataset[j]))
            M.append(Mi)
        q -= 1
    c_id = 0
    clus = [0 for i in range(len_d)]
    for k in C:
        for kk in k:
            #print(kk)
            clus[kk] = c_id
        c_id = c_id + 1
    #print(clus)
    return clus