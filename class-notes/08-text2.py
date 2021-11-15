# installs
! pip install newspaper3k
! pip install spacy
! pip install nltk
! pip install -U scikit-learn
! pip install scikit-plot
! pip install umap-learn
! pip install tokenwiser

# imports
from itertools import count
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplot

import re

# new imports
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer  
import nltk
from tokenwiser.textprep import HyphenTextPrep

from newspaper import Article

URL = "https://voicebot.ai/2021/02/16/conversational-ai-startup-admithub-raises-14m-for-higher-ed-chatbots/"

# # setup the article
article = Article(URL)

# # get the page
article.download()

# # parse it -- extracts all sorts of info
article.parse()

article.publish_date
article.text

# tokenize
cv = CountVectorizer()

# sklearn expects iterables, like lists
atext = article.text
atokens = cv.fit_transform([atext])

# how many tokens -- note the new syntax of get feature names out
len(cv.vocabulary_)
atokens.shape

# new dataset
corpus = ["tokens, tokens everywhere"]

ngrams2 = CountVectorizer(ngram_range=(1,2))
ngrams2_tok = ngrams2.fit_transform(corpus)
ngrams2.vocabulary_

ngrams3 = CountVectorizer(ngram_range=(1,3))
ngrams3.fit([atext])
ngrams3.vocabulary_
len(ngrams3.vocabulary_)
doc = ngrams3.transform([atext])

nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = list(stopwords.words('english'))
type(STOPWORDS)
STOPWORDS[:5]

stopwords.fileids()

# see how to remove stop words
cv = CountVectorizer(stop_words=STOPWORDS)
atokens = cv.fit_transform([atext])
len(cv.vocabulary_)

#CHARACTER TOKENS
x = ["Hello I can't"]
charvec = CountVectorizer(analyzer='char', ngram_range=(1,1))
char_tokens = charvec.fit(x)
charvec.vocabulary_

charvec = CountVectorizer(analyzer='char', ngram_range=(2,7))
char_tokens = charvec.fit(x)
charvec.vocabulary_

##CUSTOM PATTERNS
PATTERN = "[\w']+"
cv = CountVectorizer(token_pattern=PATTERN)
cv.fit(x)
cv.vocabulary_