import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplot

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import   KNeighborsClassifier

from sklearn.decomposition import PCA

import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer  
import nltk

from sklearn.model_selection import train_test_split
from sklearn import metrics

SQL = "SELECT * from `questrom.SMSspam.train`"
PROJECT = "ba820-fall21"
ds_train = pd.read_gbq(SQL, PROJECT)

SQL_test = "SELECT * from `questrom.SMSspam.test`"
PROJECT_test = "ba820-fall21"
ds_test = pd.read_gbq(SQL_test, PROJECT_test)

ds_train.shape
ds_train.sample(3)

cv = CountVectorizer(max_features=500)
cv.fit(ds_train.message)
cv.vocabulary_
len(cv.vocabulary_)

dtm = cv.transform(ds_train.message).toarray()
pca = PCA(50)
pcs = pca.fit_transform(dtm)

pca.explained_variance_ratio_.sum()

tree = DecisionTreeClassifier(max_depth=5, min_samples_split=30, min_samples_leaf=15, random_state=820)
tree.fit(pcs, ds_train.label)

tree.score(pcs, ds_train.label)

# apply the model to the test set
test_vs = pca.transform(cv.transform(ds_test.message).toarray())
test_preds = tree.predict(test_vs)
test_preds[:5]

# build out a dataset for the submission
ds_test['label'] = test_preds
ds_test.sample(3)

# write out sample set
ds_test[['id', 'label']].to_csv('inclass-excercise.csv', index=False)