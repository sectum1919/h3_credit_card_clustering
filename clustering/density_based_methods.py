from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

def dbscan_clustering(data):
    dbscan = DBSCAN(eps=0.01)
    label = dbscan.fit_predict(data)
    return label

def optics_clustering(data):
    optics = OPTICS()
    return optics.fit_predict(data)