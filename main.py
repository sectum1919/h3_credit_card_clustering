#%%
from dataset import dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import datetime, time

# our own methods
from clustering.kmeans import kmeans_clustering

ccdata = dataset('./data/cc_general.csv')

data = ccdata.data()

methods = [
        {"func":kmeans_clustering, "name":'kmeans'},
        # {},
        # {"func":other_cluster, "name":'other_cluster'},
    ]

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
        print(f'{method["name"]} run {end-start} seconds')

        tsne = TSNE(init='random',learning_rate=200.0)
        embedding = tsne.fit_transform(data)

        plt.scatter(embedding[:,0], embedding[:,1], c=cluster_id, s=1)
        plt.savefig(f'./fig/{method["name"]}', dpi=700)
    except:
        print(f"error occurs when running {method['name']}")

#long running
#do something other