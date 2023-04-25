import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pystac_client
from xpysom import XPySom
#import stac
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import math
import pandas as pd
import time
import re
import pickle

def nan_helper(y,cloud):
    return (cloud > 7) | (cloud <= 3), lambda z: z.nonzero()[0]

def interpolate(y,cloud):
    nans, x= nan_helper(y,cloud)
    try:
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        return y
    except:
        return y
    
def parse_geometry(geometry):
    regex = r'[0-9-\.]+'
    parsed_geom = re.findall(regex, geometry)
    parsed_geom = [float(i) for i in parsed_geom]
    return max(parsed_geom[::2]),max(parsed_geom[1::2] ), min(parsed_geom[::2]), min(parsed_geom[1::2])


bbox = pd.read_csv('./OrdemProcessamentoTiles_v0.csv').sort_values(by='ORDEMFINAL')

bbox['NM_MACRORH'] = bbox['NM_MACRORH'].replace(' ','_', regex=True).str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

parameters = dict(access_token='Enter your BDC token')
service = pystac_client.Client.open('https://brazildatacube.dpi.inpe.br/stac/', parameters=parameters)


start_date="2020-01-01"
end_date="2020-12-31"

bands = ['B04','B08','B11','SCL']

for name in np.unique(bbox['NM_MACRORH']):
    parent_dir ='/scratch/ideeps/baggio.silva/work/2020'
    directory = f'{name}'
    path = os.path.join(parent_dir, directory)
    if not os.path.exists(path):
        os.makedirs(path)
for tile in range(50,100):
        start = time.time()
        parent_dir = f"/scratch/ideeps/baggio.silva/work/2020/{bbox.iloc[tile]['NM_MACRORH']}"
        directory = f"{bbox.iloc[tile]['TILE']}"
        path = os.path.join(parent_dir, directory)
        if not os.path.exists(path):
            os.makedirs(path)
        if(len(os.listdir(path)) != 6): 
            
            item_search = service.search(bbox = parse_geometry(bbox.iloc[tile]['GEOMETRY']),
                                     datetime=f"{start_date}/{end_date}",
                                     collections=["S2-SEN2COR_10_16D_STK-1"])
            item = []
            for idx in item_search.get_items():
                item.append(idx)

            cloud = []
            ts = []
            for band in bands[:-1]:
                for i in range(len(item)):
                    with rio.open(item[i].assets[bands[-1]].href) as src:
                        cloud.append(src.read(1).flatten())

                    with rio.open(item[i].assets[band].href) as src:
                        ts.append(src.read(1).flatten())
                        profile = src.profile
                        height = src.height
                        width = src.width
                        
            
            ts = np.array(ts).T
            cloud = np.array(cloud).T

            end = time.time()
            print(f"tempo de load das séries temporais {round(end-start,2)}s \n")

            start =time.time()

            [interpolate(ts[i],cloud[i]) for i in range(len(ts))]

            end =time.time()

            print(f"tempo de interpolação das séries temporais {round(end-start,2)}s \n")

            start =time.time()

            grade=20
            n = grade
            som = XPySom(20,20, ts.shape[1],
                        random_seed=123, n_parallel=0)
            som.train(ts[::2], 40)

            end =time.time()

            print(f"tempo de treinamento das séries temporais {round(end-start,2)}s \n")

            start =time.time()

            predictions = som.winner(ts)

            position=[str((i,j)) for i in range(grade) for j in range(grade)]
            position = {pos:position.index(pos) for pos in position}
            for i in range(len(predictions)):
                predictions[i] = position[f'{predictions[i]}']
            predictions = np.array(predictions).astype("int16")
            predictions = predictions.reshape(height,width)
            
            with rio.open(path+f"/{bbox.iloc[tile]['TILE']}_som_result.tif", 'w', **profile) as dst:
                dst.write(predictions, indexes=1)

            with open(path+f"/{bbox.iloc[tile]['TILE']}_som.pickle",'wb') as handle:
                 pickle.dump(som,handle,protocol=pickle.HIGHEST_PROTOCOL)

            neuron_weights = [som.get_weights()[i][j] for i in range(n) for j in range(n)]
            
            np.save(path+f"/{bbox.iloc[tile]['TILE']}_wheights.npy",neuron_weights)

            nclust = 14
            th = 3000
            #cmap = plt.get_cmap('tab20b', nclust)
            colors = ['#ffa200','#006305' ,'#ffff00']
            (fig, ax) = plt.subplots(figsize=(18, 8), sharey=True)
            X = np.array(neuron_weights)
            sch.set_link_color_palette(colors)
            dendrogram = sch.dendrogram(sch.linkage(X, method='average'),color_threshold=th,leaf_font_size=7)
            model = AgglomerativeClustering(n_clusters=nclust, affinity='euclidean', linkage='average')#,distance_threshold=th)
            model.fit(X)
            labels = model.labels_

            plt.xlabel('Neurons CodeBook Number', fontsize=16)
            ax.spines['left'].set_lw(1.5)
            ax.spines['bottom'].set_lw(1.5)
            ax.spines['right'].set_lw(1.5)
            ax.spines['top'].set_lw(1.5)

            cmap = plt.get_cmap('tab20b', nclust)

            mlabels = labels.reshape(n,n)
            (fig, ax) = plt.subplots(figsize=(9, 9), sharey=True)
            for i in range(n):
                for j in range(n):
                    neuron_index = i * n + j
                    ax.text(i, j, str(mlabels[i][j]), va='center', ha='center')
            ax.imshow(np.array(mlabels).T, cmap=cmap, interpolation='none')
            ax.grid(True)
            plt.xticks(range(n))
            plt.yticks(range(n))
            ax.spines['left'].set_lw(1.5)
            ax.spines['bottom'].set_lw(1.5)
            ax.spines['right'].set_lw(1.5)
            ax.spines['top'].set_lw(1.5)
            arv = path+f"/{bbox.iloc[tile]['TILE']}_cluster_map"
            plt.savefig(f"{arv}.png",bbox_inches='tight')

            fig, axs = plt.subplots(n,n,figsize=(20,12) ,sharey=True)
            for neuron_index,neuron in enumerate(neuron_weights):
                col1 = math.floor(neuron_index/n)
                row1 = neuron_index % n
                axs[row1, col1].plot(neuron/10000,linewidth=2,color='black')

            
                axs[row1, col1].spines['left'].set_lw(1.5)
                axs[row1, col1].spines['bottom'].set_lw(1.5)
                axs[row1, col1].spines['right'].set_lw(1.5)
                axs[row1, col1].spines['top'].set_lw(1.5)
                axs[row1, col1].set_facecolor(cmap.colors[mlabels[col1][row1]])    


            arv = path+f"/{bbox.iloc[tile]['TILE']}_coodebook"
            plt.savefig(f"{arv}.png",bbox_inches='tight')

            dendrogram = predictions.copy()
            for i in range(len(labels)):
                    dendrogram = np.where(predictions == i, labels[i], dendrogram )
            image1 = dendrogram.reshape(height,width).astype("int16")

            
            with rio.open(path+f"/{bbox.iloc[tile]['TILE']}_14_clust.tif", 'w', **profile) as dst:
                dst.write(image1, indexes=1)

            end =time.time()
            print(f"tempo de predic + hclust das séries temporais {round(end-start,2)}s \n")
        else:
            print(f"O Tile {bbox.iloc[tile]['TILE']} já processou!! \n")
            continue
        
