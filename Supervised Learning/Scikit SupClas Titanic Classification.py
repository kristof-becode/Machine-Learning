import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the classes for Linear and Logistic Regression
from sklearn.linear_model import LinearRegression, LogisticRegression
# Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier
# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# Import svm model
from sklearn import svm
# Import Naive_Bayes model
from sklearn.naive_bayes import GaussianNB
# For Chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Import preprocessing for LabelEncoder en OneHotEncoder
from sklearn import preprocessing
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy, precision and recall calculation
from sklearn import metrics

# Display options dataframes
pd.set_option('display.width',400)
pd.set_option('display.max_columns', 40)
# Display options numpy arrays
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180

# Retrieved UCI Dataset for Mushroom Classification from https://www.kaggle.com/uciml/mushroom-classification/data
titan = pd.read_csv("/home/becode/data/pandas/train.csv")

print(titan.head(10))
# Observation : class = the label/target with values p(oisonous) and e(dible),
# the other 22 columns are possible features
# Observation : all the data is discrete and categorical, nominal but not ordinal
print(f"\nSome stats:\n ",titan.describe())
# Observation : all the columns are discrete values => take into account for Classifier selection

print(f"\nshape dataframe: {titan.shape}")
print("\nChecking for NA values:\n",titan.isna().sum())
titan = titan.drop(columns=['PassengerId','Name','Parch','Cabin','Ticket'])
#avg_age_male = titan.age[titan['sex'] =='Male'].mean
for_age_mean = titan['Age'][titan['Age'].notna()]
print(for_age_mean)
for_age_mean = titan.Age.notna()
print(for_age_mean)
print(titan.Age[titan['Sex'] == 'male'])
#for_age_mean = titan.Age[titan.Age.notna() == True & titan['Sex'] == 'male']
print(for_age_mean)
#print(titan.Age[titan['Sex'] =='Male' & titan.Age.isna() is False].mean())
print(titan.head(10))
"""
print(f"\n{mush[mush['class'] == 'p']['class'].count()} rows 'poisonous' on total {mush.shape[0]} rows -> "
      f" {round(((mush[mush['class'] == 'p']['class'].count()) / (mush.shape[0])) *100,2)}%")
print(f"{mush[mush['class'] == 'e']['class'].count()} rows 'edible' on total {mush.shape[0]} rows -> "
      f" {round(((mush[mush['class'] == 'e']['class'].count()) / (mush.shape[0])) *100,2)}%")
print("\nChecking for NA values:\n",mush.isna().sum()) # check for NA values in dataset
# Observation: No NA values in dataset
"""