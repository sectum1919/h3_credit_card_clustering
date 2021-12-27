from dataset import dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
from evaluate import unsupervised_evaluate
# our own methods
from clustering.partition_based_methods import kmeans_clustering
from clustering.density_based_methods import dbscan_clustering
from clustering.density_based_methods import optics_clustering
from clustering.hierarchical_based_methods import agglomerative_clustering

ccdata = dataset('./data/cc_general.csv')

data = ccdata.data(pca_dim=6)

methods = [
        {"func":kmeans_clustering, "name":'kmeans'},
        {"func":dbscan_clustering, "name":'dbscan'},
        {"func":optics_clustering, "name":'optics'},
        {"func":agglomerative_clustering, "name":'agglomerative'},
        # {},
        # {"func":other_cluster, "name":'other_cluster'},
    ]

tsne = TSNE(init='random',learning_rate=200.0)
embedding = tsne.fit_transform(data)

for method in methods:
    if "func" not in method:
        print("this function have no implement or you didn't add it")
        print("skipping now...")
        continue
    if "name" not in method:
        print("this function have no name, please add name")
        print("skipping now...")
        continue
    try:
        start = time.time()
        cluster_id = method["func"](data)
        end = time.time()
        print()
        print(f'{method["name"]} run {end-start} seconds')
        # do evaluate
        unsupervised_evaluate(data, cluster_id)

        plt.scatter(embedding[:,0], embedding[:,1], c=cluster_id, s=1, cmap='rainbow')
        plt.savefig(f'./fig/{method["name"]}', dpi=700)
        print()
    except:
        print(f"error occurs when running {method['name']}")

#long running
#do something other