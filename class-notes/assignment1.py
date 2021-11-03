import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.decomposition import PCA

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

scaler = StandardScaler() #standardizing the values in order to get more accurate results
scaler.fit(forums_num)
forum_scaled = scaler.transform(forums_num)

#Hierarchical Clustering
hc1 = linkage(forum_scaled, method="complete")

plt.figure(figsize=(15,5))
dendrogram(hc1, labels=forums_num.index)
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

## Compute DBSCAN
db = DBSCAN(eps=0.5, min_samples=5).fit(forums_num)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

### Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

#PCA
fcor = forums_num.corr()
sns.heatmap(fcor, cmap='Reds', center=0)
plt.show()

pca = PCA()
pcs = pca.fit_transform(forum_scaled)
type(pcs)
pcs.shape

varexp = pca.explained_variance_ratio_
type(varexp)
varexp.shape
np.sum(varexp)

#plotting variance explained by PC
plt.title("Explained Variance per PC")
sns.lineplot(range(1, len(varexp)+1), varexp)
plt.show()

#cumulative running percentage
plt.title("Explained Variance per PC")
sns.lineplot(range(1, len(varexp)+1), np.cumsum(varexp))
plt.axhline(.95)
plt.show()

explvar = pca.explained_variance_
plt.title("Eigenvalue")
sns.lineplot(range(1, len(explvar)+1), explvar)
plt.axhline(1)
plt.show()