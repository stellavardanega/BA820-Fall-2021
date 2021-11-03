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
forums.isnull().sum().sum()

forums_num = forums.select_dtypes("number") #selecting only the numerical columns from the dataset
forums_num.describe().T

#Hierarchical Clustering
hc_comp = linkage(forums_num, method="complete")
hc_sing = linkage(forums_num, method="single")
hc_avg = linkage(forums_num, method="average")
hc_ward = linkage(forums_num, method="ward")

fig, ax = plt.subplots(ncols=4, figsize=(15,5))
dendrogram(hc_comp, labels=forums_num.index, ax=ax[0])
dendrogram(hc_sing, labels=forums_num.index, ax=ax[1])
dendrogram(hc_avg, labels=forums_num.index, ax=ax[2])
dendrogram(hc_ward, labels=forums_num.index, ax=ax[3])
plt.show()

dendrogram(hc_ward, labels=forums_num.index)
plt.show()

cluster = fcluster(hc_ward, 3, criterion='maxclust')
forums['cluster'] = cluster
forums.head()

forums[forums['cluster']==1]['text']

#KMeans
k_range = range(2,15)
eval = []
silo_score = []

for i in k_range:
    k = KMeans(i)
    labs = k.fit_predict(forums_num)
    eval.append(k.inertia_)
    silo_score.append(metrics.silhouette_score(forums_num, k.predict(forums_num)))

sns.lineplot(y=eval, x=k_range)
plt.show()

sns.lineplot(y=silo_score, x=k_range)
plt.show()

k4 = KMeans(4)
k4.fit(forums_num) 
labs = k4.predict(forums_num)
forums['k4'] = labs

silo_overall = metrics.silhouette_score(forums_num, k4.predict(forums_num))
silo_overall

skplt.metrics.plot_silhouette(forums_num, labs, figsize=(7,7))
plt.show()
forums.groupby('k4').mean()