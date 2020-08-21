import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import Clustering
from sklearn.cluster import KMeans, DBSCAN
# Import preprocessing for LabelEncoder en OneHotEncoder
from sklearn import preprocessing
#Import scikit-learn metrics module
from sklearn.metrics import adjusted_rand_score

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
