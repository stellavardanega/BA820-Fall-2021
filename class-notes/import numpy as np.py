import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist, squareform

from sklearn import metrics 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import scikitplot as skplt

songs_new = pd.read_csv("~/Downloads/artist-music-catalog.csv")
songs_new.shape
songs_new.head(3)
songs_num = songs_new.iloc[:, 5:]
songs_num

sc = StandardScaler()
songs_scaled = sc.fit_transform(songs_num)
songs = pd.DataFrame(songs_scaled, columns=songs_num.columns)

diste = pdist(songs.values)
distc = pdist(songs.values, metric="cosine")

hclust_e = linkage(diste)
hclust_c = linkage(distc)

LINKS = [hclust_e, hclust_c]
TITLE = ['Euclidean', 'Cosine']

plt.figure(figsize=(15, 5))

# loop and build our plot
for i, m in enumerate(LINKS):
  plt.subplot(1, 2, i+1)
  plt.title(TITLE[i])
  dendrogram(m,
            #  labels = ps.index,
             leaf_rotation=90,
            #  leaf_font_size=10,
             orientation="left")
  
plt.show()

METHODS = ['single', 'complete', 'average', 'ward']
plt.figure(figsize=(20,5))


# loop and build our plot
for i, m in enumerate(METHODS):
  plt.subplot(1, 4, i+1)
  plt.title(m)
  dendrogram(linkage(distc, method=m), 
             leaf_rotation=90)
  
plt.show()

cluster = fcluster(hclust_c, 8, criterion='maxclust')
songs_new['cluster'] = cluster
songs_new.head()

k_range = range(2,15)
eval = []
silo_score = []

for i in k_range:
    k = KMeans(i)
    labs = k.fit_predict(songs)
    eval.append(k.inertia_)
    silo_score.append(metrics.silhouette_score(songs, k.predict(songs)))

sns.lineplot(y=eval, x=k_range)
plt.axvline(x = 4, color = 'r', linestyle = '--')
plt.text(x=2.5, y=1900, s='K = 4')
plt.axvline(x = 5, color = 'b', linestyle = '--')
plt.text(x=5.5, y=1900, s='K = 5')
plt.xlabel('Number of Clusters K')
plt.ylabel('Inertia')
plt.show()

sns.lineplot(y=silo_score, x=k_range)
plt.axvline(x = 4, color = 'r', linestyle = '--')
plt.text(x=4.5, y=0.3, s='K = 4')
plt.xlabel('Number of Clusters K')
plt.ylabel('Overall Silhouette Score')
plt.show()

k5 = KMeans(5)
k5.fit(songs) 
labs = k5.predict(songs)
songs['k5'] = labs

skplt.metrics.plot_silhouette(songs, labs, figsize=(7,7))
plt.title('Silhouette Score (K=5)')
plt.show()

tsne = TSNE()
tsne.fit(songs)

te = tsne.embedding_
te.shape

tdata = pd.DataFrame(te, columns=['e1', 'e2'])
tdata['k5'] = labs

plt.figure(figsize=(10, 8))
sns.scatterplot(x="e1", y="e2", hue='k5', data=tdata, legend="full")
plt.show()

songs.groupby('k5').count()['tempo']
songs_new.groupby('cluster').count()['tempo']

