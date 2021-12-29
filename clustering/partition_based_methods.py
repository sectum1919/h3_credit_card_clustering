from sklearn.cluster import KMeans
import numpy as np
np.random.seed(20211227)
def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    X = X.reshape(X.shape[0], -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances



class Kmeans():
    def __init__(self, k, max_iterations=500, varepsilon=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon

    def init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids
    
    def _closest_centroid(self, sample, centroids):
        distances = euclidean_distance(sample, centroids)
        closest_i = np.argmin(distances)
        return closest_i

    def create_clusters(self, centroids, X):
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    def predict(self, X):
        centroids = self.init_random_centroids(X)

        for _ in range(self.max_iterations):
            clusters = self.create_clusters(centroids, X)
            former_centroids = centroids
            
            centroids = self.update_centroids(clusters, X)
            
            diff = centroids - former_centroids
            if diff.any() < self.varepsilon:
                break
            
        return self.get_cluster_labels(clusters, X)


def kmeans_byself_clustering(data):
    clf = Kmeans(k = 3)
    return clf.predict(data)

def kmeans_clustering(data):
    k_means = KMeans(n_clusters= 3)
    k_means.fit(data)
    return k_means.predict(data)

# import matplotlib.pyplot as plt
# def kmeans_zhou(data):
#     SSE = []
#     for k in range(1, 15):
#         k_means = KMeans(n_clusters=k)
#         k_means.fit(data)
#         SSE.append(k_means.inertia_)
#         temp = k_means.inertia_
#     X = range(1, 15)
#     plt.xlabel('k')
#     plt.ylabel('SSE')
#     plt.plot(X, SSE, 'o-')
#     plt.savefig(f'./fig/kmeans_SSE', dpi=700)
#     return k_means.predict(data)