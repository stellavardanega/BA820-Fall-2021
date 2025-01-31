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
diamonds = dia.select_dtypes("number") #beware that some numbers may be categorical --> wouldn't work

#3.
diamonds.describe().T
scaler = StandardScaler()
scaler.fit(diamonds)

dia_scaled = scaler.transform(diamonds)

k5 = KMeans(5)
k5_labs = k5.fit_predict(dia_scaled)
np.unique(k5_labs)

dia['k5'] = k5_labs

#4. 
skplt.metrics.plot_silhouette(dia_scaled, k5_labs, figsize=(7,7))
plt.show()

#5. 
sns.boxplot(data=dia, x="k5", y='carat')
plt.show()

#PCA - Start of class
SQL = "select * from `questrom.datasets.judges`"
PROJECT = "ba820-fall21"
judges = pd.read_gbq(SQL, PROJECT)

judges.info()
judges.set_index('judge', inplace=True)

#correlation matrix
jcor = judges.corr()
sns.heatmap(jcor, cmap='Reds', center=0)
plt.show()

#fit our first PCA model
pca = PCA()
pcs = pca.fit_transform(judges)
type(pcs)
pcs.shape

pcs[:5, :5]

#variance explanation ratio -- pc explained variance
varexp = pca.explained_variance_ratio_
type(varexp)
varexp.shape
np.sum(varexp)

#plot the varince explained the PC
plt.title("Explained Variance per PC")
sns.lineplot(range(1, len(varexp)+1), varexp)
plt.show()

#cumulative running percentage
plt.title("Explained Variance per PC")
sns.lineplot(range(1, len(varexp)+1), np.cumsum(varexp))
plt.axhline(.95)
plt.show()

#explained variance - eigenvalue
#other ways to summarize PCs
explvar = pca.explained_variance_
plt.title("Eigenvalue")
sns.lineplot(range(1, len(explvar)+1), explvar)
plt.axhline(1)
plt.show()

#EXERCISE
# fit a pca model for the diamonds
# make sure you think abotu data pre-processing
# plot out the explained variance of the new PCAs based on your dataset

SQL = "SELECT * from `questrom.datasets.diamonds`"
PROJECT = "ba820-fall21"
dia = pd.read_gbq(SQL, PROJECT)

dia.head(3)
diamonds = dia.select_dtypes("number")
diamonds.head(3)

scaler = StandardScaler()
scaler.fit(diamonds)
dia_scaled = scaler.transform(diamonds)

pca_dia = PCA()
pcs_dia = pca_dia.fit_transform(dia_scaled)

#plot % variance explained per PC for the diamon dataset
explvar_dia = pca_dia.explained_variance_ratio_
plt.title("% of Variance Explained per PC - Diamonds")
sns.lineplot(range(1, len(explvar_dia)+1), explvar_dia)
plt.show()

#### judges
pca.n_components_

COLS = ['PC' + str(i) for i in range(1, len(varexp)+1)]
COLS

comps = pca.components_
loadings = pd.DataFrame(comps.T, columns=COLS, index=judges.columns)
loadings.head()

sns.heatmap(loadings, cmap='vlag')
plt.show()

pcs[:5, :5]

j = pd.DataFrame(pcs[:, :2], columns=['pc1', 'pc2'], index=judges.index)
sns.scatterplot(data=j, x='pc1', y='pc2')
plt.show()