import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import Clustering
from sklearn.cluster import KMeans, DBSCAN
# Import preprocessing for LabelEncoder en OneHotEncoder
from sklearn import preprocessing
# Import scikit-learn metrics module
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Display options dataframes
pd.set_option('display.width',400)
pd.set_option('display.max_columns', 40)
# Display options numpy arrays
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180

credit = pd.read_csv("/home/becode/LearnAI/scikit/scikit/Unsupervised Learning/datasets_14701_19663_CC GENERAL.csv")
print(credit.head(10))
print(credit.shape)
print(credit.describe())
print(credit.info())
print(credit.isna().sum())
# replace NA's in MINIMUM_PAYMENTS and CREDIT_LIMIT, respectively 10 and 1 NA's and check afterwards
credit.MINIMUM_PAYMENTS.fillna(credit.MINIMUM_PAYMENTS.mean(), inplace=True)
credit.CREDIT_LIMIT.fillna(credit.CREDIT_LIMIT.mean(),inplace=True)
print(credit.isna().sum())

# Customer ID is irrelevant for clustering users
credit = credit.drop('CUST_ID', axis =1)

# print correlation heat map (Pearson's coeff)
"""
corr=credit.corr()
top_features=corr.index
plt.figure(figsize=(20,20))
sns.heatmap(credit[top_features].corr(),annot=True)
plt.show()
"""
# Look at data spread
sns.boxplot(data=credit)
plt.show()

# Standardizing the data
X = credit # or X = np.asarray(credit)
stan = preprocessing.StandardScaler()
X = stan.fit_transform(X)

"""
# Elbow curve : inertia vs k
n_clusters=20
cost=[]
for i in range(1,n_clusters):
    kmeans= KMeans(i)
    kmeans.fit(X)
    cost.append(kmeans.inertia_)
plt.plot(range(1,20),cost)
plt.xticks(range(1,20,2))
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
"""

# Silhouette curve : Silhouette vs k
"""
The silhouette coefficient can vary between -1 and +1: a coefficient close to +1 means that the instance is well 
inside its own cluster and far from other clusters, while a coefficient close to 0 means that it is close to a 
cluster boundary, and finally a coefficient close to -1 means that the instance may have been assigned to the wrong
cluster.
"""
"""
n_clusters=20
sil_scores=[]
for i in range(2,n_clusters):  # n_clusters can not be 1, took me a really long time to change the range from range(1, n) tp (2,n)
    kmeans = KMeans(i)
    labels = kmeans.fit_predict(X) # or kmeans.labels_ is the same
    sil_scores.append(silhouette_score(X,labels))
plt.plot(range(2,20),sil_scores)
plt.title('The Silhouette curve')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette')
plt.show()
"""
# Silhouette scores highest for k=3 but only marginal differnce, the score is too close to 0, 1 means good clustering

# Call KMeans and calculate silhouette score, can't calculate adjsuetd rand 'cause I don't have an actual y)
# Only 3 clusters
kmeans = KMeans(n_clusters=3, init='k-means++',n_init=10, max_iter=300, random_state=123 )
kmeans.fit_predict(X)
print(f"silhouette score = {silhouette_score(X,kmeans.labels_)}")
# I think I really need to take a look at the data and reorganize it, arrange the outliers !

# Facetgrid Plots per clusters:
"""
clusters = credit
clusters['Clusters'] = kmeans.labels_  # add column with cluster labels to a copy of our dataframe
print(clusters.head(10))
i=0
fig1 = plt.figure()
## number of co??
for column in clusters:
    i += 1
    grid = sns.FacetGrid(clusters, col='Clusters')
    grid.map(plt.hist,column)
    ax = fig1.add_subplot(len(clusters.columns.tolist()),1, i)
plt.show()
"""

# PCA to visualize clusters, my clustering is poor
dist = 1 - cosine_similarity(X)

pca = PCA(2)
pca.fit(dist)
X_PCA = pca.transform(dist)
print(X_PCA.shape)
x, y = X_PCA[:, 0], X_PCA[:, 1]

labels = kmeans.labels_

colors = {0: 'red',
          1: 'blue',
          2: 'green'}

names = {0: 'who make all type of purchases',
         1: 'more people with due payments',
         2: 'who purchases mostly in installments'}

df = pd.DataFrame({'x': x, 'y': y, 'label': labels})
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(20, 13))

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,
            color=colors[name], label=names[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')

ax.legend()
ax.set_title("Customers Segmentation based on their Credit Card usage bhaviour.")
plt.show()





"""
columns = ['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT',
           'PAYMENTS', 'MINIMUM_PAYMENTS']

for c in columns:
    Range = c + '_RANGE'
    data[Range] = 0
    data.loc[((data[c] > 0) & (data[c] <= 500)), Range] = 1
    data.loc[((data[c] > 500) & (data[c] <= 1000)), Range] = 2
    data.loc[((data[c] > 1000) & (data[c] <= 3000)), Range] = 3
    data.loc[((data[c] > 3000) & (data[c] <= 5000)), Range] = 4
    data.loc[((data[c] > 5000) & (data[c] <= 10000)), Range] = 5
    data.loc[((data[c] > 10000)), Range] = 6

columns = ['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
           'CASH_ADVANCE_FREQUENCY', 'PRC_FULL_PAYMENT']

for c in columns:
    Range = c + '_RANGE'
    data[Range] = 0
    data.loc[((data[c] > 0) & (data[c] <= 0.1)), Range] = 1
    data.loc[((data[c] > 0.1) & (data[c] <= 0.2)), Range] = 2
    data.loc[((data[c] > 0.2) & (data[c] <= 0.3)), Range] = 3
    data.loc[((data[c] > 0.3) & (data[c] <= 0.4)), Range] = 4
    data.loc[((data[c] > 0.4) & (data[c] <= 0.5)), Range] = 5
    data.loc[((data[c] > 0.5) & (data[c] <= 0.6)), Range] = 6
    data.loc[((data[c] > 0.6) & (data[c] <= 0.7)), Range] = 7
    data.loc[((data[c] > 0.7) & (data[c] <= 0.8)), Range] = 8
    data.loc[((data[c] > 0.8) & (data[c] <= 0.9)), Range] = 9
    data.loc[((data[c] > 0.9) & (data[c] <= 1.0)), Range] = 10

columns = ['PURCHASES_TRX', 'CASH_ADVANCE_TRX']

for c in columns:
    Range = c + '_RANGE'
    data[Range] = 0
    data.loc[((data[c] > 0) & (data[c] <= 5)), Range] = 1
    data.loc[((data[c] > 5) & (data[c] <= 10)), Range] = 2
    data.loc[((data[c] > 10) & (data[c] <= 15)), Range] = 3
    data.loc[((data[c] > 15) & (data[c] <= 20)), Range] = 4
    data.loc[((data[c] > 20) & (data[c] <= 30)), Range] = 5
    data.loc[((data[c] > 30) & (data[c] <= 50)), Range] = 6
    data.loc[((data[c] > 50) & (data[c] <= 100)), Range] = 7
    data.loc[((data[c] > 100)), Range] = 8
"""
# from sklearn.metrics import silhouette_score
# silhouette_score(X, kmeans.labels_)