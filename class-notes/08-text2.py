# installs
! pip install newspaper3k
! pip install spacy
! pip install nltk
! pip install -U scikit-learn
! pip install scikit-plot
! pip install umap-learn
! pip install tokenwiser

# imports
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