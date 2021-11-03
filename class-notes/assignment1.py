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

scaler = StandardScaler() #standardizing the values in order to get more accurate results
scaler.fit(forums_num)
forum_scaled = scaler.transform(forums_num)

#Hierarchical Clustering
hc1 = linkage(forum_scaled, method="complete")

plt.figure(figsize=(15,5))
dendrogram(hc1, labels=forums_num.index)
plt.show()

cluster = fcluster(hc1, 6, criterion="maxclust")
forums['cluster'] = cluster
forums.head()

forums[(forums['cluster']==3)]['text']