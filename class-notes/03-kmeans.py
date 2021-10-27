# WARMUP EXERICSE:
# dataset:
# https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Election08.csv

# task
# use hierarchical clustering on the election dataset
# keep just the numerical columns
# add state abbreviation as the index
# use complete linkage and generate 4 clusters
# put back onto the original dataset
# profile the number of states by cluster assignment and the % that Obama won

# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# what we need for today
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

from sklearn import metrics 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import scikitplot as skplt

eo = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Election08.csv')
eo.shape
eo.head(3)

# state an index
# just numeric data
eo.columns = eo.columns.str.lower()
eo.set_index('abr', inplace=True)

election = eo.loc[:, "income":"dem.rep"] #selecting only numerical columns and assigning to new dataset

#standardize
scaler = StandardScaler()
escaled = scaler.fit_transform(election)
type(escaled)

#cluster
hc1 = linkage(escaled, method="complete")
hc1

# create the plot
plt.figure(figsize=(15,5))
dendrogram(hc1, labels=election.index)
plt.show()

# create 4 clusters
cluster = fcluster(hc1, 4, criterion="maxclust")
cluster

eo['cluster'] = cluster
eo.head(3)

# simple profile of a cluster
eo.groupby('cluster')['obamawin'].mean()
eo.cluster.value_counts()

eo.loc[eo.cluster==4, :]

#KMEANS - Start of Lesson
SQL = "SELECT * from `questrom.datasets.judges`"
PROJECT = "ba820-fall21"
judges = pd.read_gbq(SQL, PROJECT)

judges.shape
judges.sample(3)

#data cleanup
#clean column names
#judge index -- numeric
judges.set_index('judge', inplace=True)
judges.columns = judges.columns.str.lower()

judges.describe().T

# fit our first kmean cluster
k3 = KMeans(3) #initialize with number of clusters
k3.fit(judges) 
labs = k3.predict(judges)
labs

#how many iterations ran
k3.n_iter_

#put these back onto the original dataset
judges['k3'] = labs
judges.sample(3)

#do our first profile
k3_profile = judges.groupby("k3").mean()
k3_profile.T

sns.heatmap(k3_profile, cmap='Reds')
plt.show()

#exercise
#fit cluster solution with 5 clusters
#apply it back it to dataset
#do a quick profile/persona of the clusters
j = judges.drop('k3', axis = 1)
j.head(3)
k5 = KMeans(5)
k5.fit(j) 
labs2 = k5.predict(judges)
j['k5'] = labs2

k5_profile = j.groupby("k5").mean()
sns.heatmap(k5_profile, cmap='Blues')
plt.show()

k5.inertia_ #pick this one because it minimizes inertia value
k3.inertia_

#exercise
#fit range of cluster solutions for 2 to 10, k=2, k=3...
#save out a way to evaluate the solutions based on the inertia of the fit
k_range = range(2,11)
eval = []

for i in k_range:
    k = KMeans(i)
    labs = k.fit_predict(j)
    eval.append(k.inertia_)

sns.lineplot(y=eval, x=k_range)
plt.show()

