import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

from sklearn import metrics 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import scikitplot as skplt

#Importing and preparing the dataset
forums = pd.read_pickle("~/Documents/GitHub/BA820-Fall-2021/assignments/assignment-01/forums.pkl")
type(forums)
forums.shape
forums.sample(3)

forums_num = forums.select_dtypes("number") #selecting only the numerical columns from the dataset
forums_num.describe().T

#Hierarchical Clustering
hc1 = linkage(forums_num, method="complete")
hc2 = linkage(forums_num, method="single")
hc3 = linkage(forums_num, method="average")

plt.figure(figsize=(15,5))
dendrogram(list(hc1, hc2, hc3), labels=forums_num.index)
plt.show()

cluster = fcluster(hc1, 80, criterion='distance')
forums['cluster'] = cluster
forums.head()

forums[(forums['cluster']==3)]['text']

#KMeans
k_range = range(2,15)
eval = []

for i in k_range:
    k = KMeans(i)
    labs = k.fit_predict(forums_num)
    eval.append(k.inertia_)

sns.lineplot(y=eval, x=k_range)
plt.show()

k9 = KMeans(9)
k9.fit(forums_num) 
labs = k9.predict(forums_num)
forums['k9'] = labs

silo_overall = metrics.silhouette_score(forums_num, k9.predict(forums_num))
silo_overall

k4 = KMeans(4)
k4.fit(forums_num) 
labs = k4.predict(forums_num)
forums['k4'] = labs

silo_overall = metrics.silhouette_score(forums_num, k4.predict(forums_num))
silo_overall

skplt.metrics.plot_silhouette(forum_scaled, labs, figsize=(7,7))
plt.show()
forums.groupby('k4').mean()

## Compute DBSCAN
db = DBSCAN(eps=0.5, min_samples=5).fit(forums_num)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

### Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)