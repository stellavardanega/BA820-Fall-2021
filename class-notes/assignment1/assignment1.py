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

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15,5))
dendrogram(hc_comp, labels=forums_num.index, ax=ax1)
dendrogram(hc_sing, labels=forums_num.index, ax=ax2)
dendrogram(hc_avg, labels=forums_num.index, ax=ax3)
dendrogram(hc_ward, labels=forums_num.index, ax=ax4)
ax1.set_title('Complete')
ax2.set_title('Single')
ax3.set_title('Average')
ax4.set_title('Ward')
ax1.tick_params(labelbottom=False)
ax2.tick_params(labelbottom=False)
ax3.tick_params(labelbottom=False)
ax4.tick_params(labelbottom=False)
fig.supxlabel('Values Storing Message Details')
plt.show()

dendrogram(hc_ward, labels=forums_num.index)
plt.axhline(y = 17, color = 'r', linestyle = '--')
plt.axhline(y = 14, color = 'b', linestyle = '--')
plt.tick_params(bottom=False, labelbottom=False)
plt.xlabel('Values Storing Message Details')
plt.title('Dendrogram (Ward)')
plt.show()

cluster = fcluster(hc_ward, 4, criterion='maxclust')
forums['cluster'] = cluster
forums.head()

forums[forums['cluster']==4]['text']

#KMeans
forums.drop('cluster', axis=1, inplace=True)
k_range = range(2,15)
eval = []
silo_score = []

for i in k_range:
    k = KMeans(i)
    labs = k.fit_predict(forums_num)
    eval.append(k.inertia_)
    silo_score.append(metrics.silhouette_score(forums_num, k.predict(forums_num)))

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
plt.axvline(x = 3, color = 'b', linestyle = '--')
plt.text(x=1.5, y=0.09, s='K = 3')
plt.xlabel('Number of Clusters K')
plt.ylabel('Overall Silhouette Score')
plt.show()

k4 = KMeans(4)
k4.fit(forums_num) 
labs = k4.predict(forums_num)
forums['k4'] = labs

skplt.metrics.plot_silhouette(forums_num, labs, figsize=(7,7))
plt.title('Silhouette Score (K=4)')
plt.show()

forums.drop('k4', axis=1, inplace=True)
k5 = KMeans(5)
k5.fit(forums_num) 
labs_5 = k5.predict(forums_num)
forums['k5'] = labs

skplt.metrics.plot_silhouette(forums_num, labs_5, figsize=(7,7))
plt.title('Silhouette Score (K=5)')
plt.show()

forums.drop('k5', axis=1, inplace=True)
k3 = KMeans(3)
k3.fit(forums_num) 
labs_3 = k3.predict(forums_num)
forums['k3'] = labs

skplt.metrics.plot_silhouette(forums_num, labs_3, figsize=(7,7))
plt.title('Silhouette Score (K=3)')
plt.show()

forums.drop('k3', axis=1, inplace=True)
forums['k4'] = labs
summary_clusters = forums.groupby('k4').mean()
summary_clusters.to_csv(r'~/Documents/GitHub/ba820-fall-2021/class-notes/assignment1/summary-clusters.csv')
forums[forums['k4']==1]['text']

cluster1 = forums[forums['k4']==1]['text']
cluster2 = forums[forums['k4']==2]['text']
cluster3 = forums[forums['k4']==3]['text']
cluster4 = forums[forums['k4']==4]['text']
cluster1.to_csv(r'~/Documents/GitHub/ba820-fall-2021/class-notes/assignment1/cluster1.csv')
cluster2.to_csv(r'~/Documents/GitHub/ba820-fall-2021/class-notes/assignment1/cluster2.csv')
cluster3.to_csv(r'~/Documents/GitHub/ba820-fall-2021/class-notes/assignment1/cluster3.csv')
cluster4.to_csv(r'~/Documents/GitHub/ba820-fall-2021/class-notes/assignment1/cluster4.csv')