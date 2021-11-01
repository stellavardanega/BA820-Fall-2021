# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# what we need for today
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics 
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

import scikitplot as skplt

# color maps
from matplotlib import cm


# resources
# Seaborn color maps/palettes:  https://seaborn.pydata.org/tutorial/color_palettes.html
# Matplotlib color maps:  https://matplotlib.org/stable/tutorials/colors/colormaps.html
# Good discussion on loadings: https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html



##############################################################################
## Warmup Exercise
##############################################################################

# warmup exercise
# questrom.datasets.diamonds
# 1. write SQL to get the diamonds table from Big Query
# 2. keep only numeric columns (pandas can be your friend here!)
# 3. use kmeans to fit a 5 cluster solution
# 4. generate the silohouette plot for the solution
# 5. create a boxplot of the column carat by cluster label (one boxplot for each cluster)
#1.
SQL = "SELECT * from `questrom.datasets.diamonds`"
PROJECT = "ba820-fall21"
dia = pd.read_gbq(SQL, PROJECT)
dia.shape

#2.
dia.head(3)
diamonds = dia.loc[:, "depth":"z"]
diamonds['carat'] = dia[['carat']]
diamonds.head(3)

#3.
scaler = StandardScaler()
escaled = scaler.fit_transform(diamonds)
k5 = KMeans(5) #initialize with number of clusters
k5.fit(diamonds) 
labs = k5.predict(diamonds)
labs