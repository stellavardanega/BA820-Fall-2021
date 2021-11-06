# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px    # interactive plotting for notebook environments

from mlxtend.frequent_patterns import apriori, association_rules

# my BILLING project in Google Cloud - replace "questrom" with your project
PROJECT = "ba820-fall21"

# SQL
SQL = """
select * from `questrom.datasets.crm_campaign`
"""

crm = pd.read_gbq(SQL, PROJECT)

#Ex 1
type(crm)
crm.shape
crm.sample(3)

crm.drop(columns="contcat", inplace=True)
crm.drop_duplicates(inplace=True)
crm.isna().sum()
crm.dropna(inplace=True)
crm.isna().sum()

crm['flag'] = True
db = crm.pivot(index="crm_id", columns="contcode", values="flag")
db.fillna(False, inplace=True)
db.shape

#Ex 2
db.head(3)

#1.
converted = db.CMO.sum() 
print(converted)
db.CMO.mean()

#2.
interactions = db.sum(axis=0)
interactions.sort_values(ascending=False)[:10]

#3.
user_ints = db.mean(axis=1)
user_ints

#4.
interactions_f = interactions / len(db)
interactions_f.sort_values(ascending=False, inplace=True)
sns.lineplot(range(len(interactions_f)), interactions_f.values)
plt.show()

#Ex 3
itemsets = apriori(db, min_support=.0002, use_colnames=True)
rules = association_rules(itemsets, metric="support", min_threshold=.0002)

#Ex 4
rules.describe()
rules.sample(3)

#3.
rules.sort_values('lift', ascending=False).head(10)

#4.
rules['count'] = rules.support * len(db)
rules.sort_values('count', ascending=False).head(10)

#5.
rules['lhs_len'] = rules.antecedents.apply(lambda x: len(x))
rules.loc[rules.lhs_len == 6, :].shape
rules.loc[rules.lhs_len == 6, :].sample(5)

#Ex 5
strategy = rules.copy()
strategy['rhs_len'] = strategy.consequents.apply(lambda x: len(x))
ROWS = np.where((strategy.consequents=={'CMO'}) & (strategy.rhs_len == 1))
strategy = strategy.iloc[ROWS[0], :]

strategy.shape
strategy.head()

strategy.sort_values('lhs_len', ascending=False).head(10)

sns.scatterplot(data=strategy, x="support", y="confidence", hue="lift")
plt.show()

sns.boxplot(data=strategy, x="lhs_len", y="confidence", color="grey")
plt.show()

rule1 = strategy.loc[strategy.lhs_len == 1, :]
rule1.sort_values("lift", ascending=False, inplace=True)

rule1.head(10)