import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import Scikit Learn datasets
from sklearn.datasets import make_blobs, make_moons
# Import Clustering
from sklearn.cluster import KMeans, DBSCAN

from sklearn.metrics import adjusted_rand_score

# Display options dataframes
pd.set_option('display.width',400)
pd.set_option('display.max_columns', 40)
# Display options numpy arrays
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180

# Load blobs dataset from Scikit Learn
# NOTE : Loading it does not work, it is a function (to generate isotropic Gaussian blobs for clustering) so better
# just call it
# NOT THIS : data = datasets.load_make_blobs()
# NOT THIS : print (data.DESCR)

# generate 2d classification dataset
X, y = make_blobs(n_samples=500, centers=3, n_features=2)
# scatter plot, dots colored by class value
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()

# I don't think I need MInmAxScaler or normalize for this data -> maybe verify?
# For clustering you also don't split  in train and test data, because you can't train!!!!!

# define the model
model = KMeans(n_clusters = 3)
# fit model and predict clusters
y_hat = model.fit_predict(X)
# retrieve unique clusters
clusters = np.unique(y_hat)
# create scatter plot for samples from each cluster
for cluster in clusters:
# get row indexes for samples with this cluster
	row_ix = np.where(y_hat == cluster)
# create scatter of these samples
	plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
plt.show()

# ari between -1 and 1 , 0 means random clustering, 1 means perfect clustering
ari_kmeans = adjusted_rand_score(y, y_hat)
print(ari_kmeans)

"""
make_moons()
"""

# generate 2d classification dataset
X, y = make_moons(n_samples=500, noise=0.1)
# scatter plot, dots colored by class value
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()

# define the model
model = DBSCAN(eps=0.11, min_samples=7)
# fit model and predict clusters
y_hat = model.fit_predict(X)
# retrieve unique clusters
clusters = np.unique(y_hat)
# create scatter plot for samples from each cluster
for cluster in clusters:
# get row indexes for samples with this cluster
	row_ix = np.where(y_hat == cluster)
# create scatter of these samples
	plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
plt.show()

ari_dbscan = adjusted_rand_score(y, y_hat)
print(ari_dbscan)

"""
load_digits

https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html#sphx-glr-auto-examples-datasets-plot-digits-last-image-py
"""
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.DESCR)
print(digits.data.shape)
# Have a look at the data
df = pd.DataFrame(digits.data)
print(df.head(10))
target = pd.DataFrame(digits.target)
print(target)

# Display digit FIRST WAY
plt.gray() 
#plt.matshow(digits.images[0])
#plt.show()
# Display digit alternative
plt.figure(1, figsize=(5,5))
#plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
#plt.show()

X = digits.data
y = digits.target
# define the model
model = KMeans(n_clusters = 10, max_iter=500, algorithm='full') # results stay around 67% despite playing with params
# fit model and predict clusters
y_hat = model.fit_predict(X)
# retrieve unique clusters
clusters = np.unique(y_hat)
# create scatter plot for samples from each cluster
for cluster in clusters:
# get row indexes for samples with this cluster
	row_ix = np.where(y_hat == cluster)
# create scatter of these samples
	plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
plt.show()

# ari between -1 and 1 , 0 means random clustering, 1 means perfect clustering
ari_kmeans = adjusted_rand_score(y, y_hat)
print(ari_kmeans)
"""
IRIS
"""
from sklearn import datasets
iris = datasets.load_iris()
print(iris.DESCR)
print(iris.data)
print(iris.target)
data = pd.DataFrame(iris.data)
target = pd.DataFrame(iris.target)
print(data.head(10))
print(target.head(10))
# In target there are 3 classes; one for each species of Iris, so 3 clusters

X = iris.data
y = iris.target
# define the model
model = KMeans(n_clusters = 3)
# fit model and predict clusters
y_hat = model.fit_predict(X)
# retrieve unique clusters
clusters = np.unique(y_hat)
# create scatter plot for samples from each cluster
for cluster in clusters:
# get row indexes for samples with this cluster
	row_ix = np.where(y_hat == cluster)
	plt.scatter(X[row_ix, 0], X[row_ix, 1])
plt.show()
# ari between -1 and 1 , 0 means random clustering, 1 means perfect clustering
ari_kmeans = adjusted_rand_score(y, y_hat)
print(ari_kmeans)

# AS A FUNCTION
def plant_clustering(n_clusters : int) -> float :
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    model = KMeans(n_clusters)
    y_hat = model.fit_predict(X)
    ari_kmeans = adjusted_rand_score(y, y_hat)
    return ari_kmeans

print(plant_clustering(3))

## REMARKS : You can' t train in Unsupervised or Clustering but if you have a label/target column then you can still
## evaluate the clustering with ARI or Silhouette
