import matplotlib.pyplot as plt
import seaborn
import pandas as pd

def draw_grid(df, label, name):
    clusters = pd.concat([df, pd.DataFrame({'cluster':label})], axis=1)
    for c in clusters:
        grid = seaborn.FacetGrid(clusters, col='cluster')
        grid.map(plt.hist, c)
        grid.savefig(f'./fig/{name}_result_{c}', dpi=700)
        plt.close(grid.fig)
        ans = []
        for i in range(0,3):
            temp = clusters.loc[clusters.cluster == i].loc[:,c]
            ans.append(round(temp.sum(axis=0)/temp.shape[0],2))
        
        print(f'{c},{ans[0]},{ans[1]},{ans[2]}')