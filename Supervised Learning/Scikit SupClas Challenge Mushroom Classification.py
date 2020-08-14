import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Import the classes for Linear and Logistic Regression
from sklearn.linear_model import LinearRegression, LogisticRegression
#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier
# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
#Import svm model
from sklearn import svm
#Import Naive_Bayes model
from sklearn.naive_bayes import GaussianNB

# Import preprocessing for LabelEncoder
from sklearn import preprocessing
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

#Display options dataframes
pd.set_option('display.width',400)
pd.set_option('display.max_columns', 40)

# Retrieved UCI Dataset for Mushroom Classification from https://www.kaggle.com/uciml/mushroom-classification/data
mush= pd.read_csv("/home/becode/LearnAI/scikit/scikit/Supervised Learning/datasets_478_974_mushrooms.csv")

print(mush.head(10))
# class = the label with p(oisonous) and e(dible) , the other 22 columns are possible features
print(mush.describe()) # Observation : all the columns are discrete values => take into account for Classifier selection
print(mush.shape)
print(mush.isna().sum()) # check for NA values in dataset, there are none apparently
# Observation1 : all the columns are discrete values => take into account for Classifier selection, I think all the data
# categorical but not ordinal

# Obs 2 : veil-type has unique value 'p' so I can drop that column
mush=mush.drop('veil-type', axis=1) # axis 1 means drop column, axis= 0 is row


# Encode all colmuns beside target /label column 'class'
lab_enc = preprocessing.LabelEncoder()
#xx = mush.drop('class', axis=1)
x = lab_enc.fit_transform(mush.odor) # label encode possible features and take out label 'class'
X = np.reshape(x,(-1, 1))
print(X)

# LabelEncode each column -> caveat, shouldn't I be using OneHotEncoder, LabelEncode should only be for label
#for column in mush.drop('class', axis=1).columns: # I first skipped my label/target but I need to include it too
for column in mush.columns:
    mush[column] = lab_enc.fit_transform(mush[column])
    #print(mush[column])
print(mush.head(10))
# !!!!!!! (p)oisonous = 1, (e)dible = 0 !!!!!!

# check which columns or features to drop
# Test Lin Regfor individual features
relation = []
rsquared = []
for column in mush.drop('class',axis=1).columns:
    x = mush[column].values
    X = np.reshape(x,(-1, 1))
    y = mush['class'].values
    model = LinearRegression().fit(X, y)
    relation.append(f"Class vs {column}")
    rsquared.append(model.score(X, y))
    #print(f"R squared for class vs {column}:{model.score(X, y)}")
r_scores = pd.DataFrame({'Relation': relation, 'R²':rsquared})
print(r_scores.sort_values(by='R²', ascending=False))

# Test Lin Reg for selection of features
X = mush[['gill-size','gill-color','bruises']]
#X = np.reshape(x, (-1, 1))
y = mush['class'].values
model = LinearRegression().fit(X, y)
print(f"R squared for class vs 'gill-size','gill-color','bruises':{model.score(X, y)}")

# Test Lin Reg for selection of features
X = mush[['gill-size','gill-color','bruises','ring-type']]
#X = np.reshape(x, (-1, 1))
y = mush['class'].values
model = LinearRegression().fit(X, y)
print(f"R squared for class vs 'gill-size','gill-color','bruises','ring-type':{model.score(X, y)}")

# Test Lin Reg for selection of features
X = mush[['gill-size','gill-color','bruises','ring-type','stalk-root']]
#X = np.reshape(x, (-1, 1))
y = mush['class'].values
model = LinearRegression().fit(X, y)
print(f"R squared for class vs 'gill-size','gill-color','bruises','ring-type','stalk-root':{model.score(X, y)}")

# Test Lin Reg for all features combined THIS SCORES THE BEST REGESSION
X = mush.drop('class', axis=1)
#X = np.reshape(x, (-1, 1))
y = mush['class'].values
model = LinearRegression().fit(X, y)
print(f"R squared for class vs all columns:{model.score(X, y)}")

"""
# PLOTS
for column in mush.drop('class', axis=1).columns:
    x = mush[column].values
    X = np.reshape(x,(-1, 1))
    y = mush['class'].values
    # PLOT
    #p = df.columns.tolist().index(item) + 1
    #plt.figure(1, figsize=(15, 15))
    #plt.subplot(2, 5, p)  # 2 rows, 5 columns, 1st subplot = top left
    plt.scatter(X, y, s=0.75)
    plt.ylabel("class")
    plt.xlabel(column)
    plt.show()
#path = "/home/becode/LearnAI/scikit/Diabetes_plots.png"
#plt.savefig(path, transparent=False)
#plt.show()
"""

# TRY KNN for different K-VALUES AND NORMALIZED DATASET
print("\n-> KNN\n")
X = mush.drop('class', axis=1)
#print(X)
X_normalized = preprocessing.normalize(X, norm='l2')
y = mush['class'].values
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3,random_state=123) # 70% training and 30% test
k_list = [1,3,5,9]
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"KNN accuracy for K={k} with normalization : {metrics.accuracy_score(y_test, y_pred)}")

# TRY LOGISTIC REGRESSION
print("\n-> LOGISTIC REGRESSION\n")
X = mush.drop('class', axis=1)
stan = preprocessing.StandardScaler()
X_standardized = stan.fit_transform(X)
#X_standardized = preprocessing.StandardScaler(X)
y = mush['class'].values
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3,random_state=123) # 70% training and 30% test
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print("\nLogistic regression accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Logistic regression precision:",metrics.precision_score(y_test, y_pred))
print("Logistic regression recall:",metrics.recall_score(y_test, y_pred))

# TRY DECISION TREE , NO SCALING NEEDED
print("\n-> DECISION TREE\n")
X = mush.drop('class', axis=1)
y = mush['class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123) # 70% training and 30% test
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

criteria={'criterion':['gini', 'entropy'],'split' :['best', 'random'], 'depth':[None, 1,2,3,4,5]}
for criterion in criteria['criterion']:
    for splitter in criteria['split']:
        for depth in criteria['depth']:
            clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=depth)
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(f"Accuracy for criterion {criterion}, splitter {splitter} and max_depth {depth} : {metrics.accuracy_score(y_test, y_pred)}")

# TRY SVM # NORMALIZATION GOOD IDEA
print("\n-> SVM\n")
X = mush.drop('class', axis=1)
X_normalized = preprocessing.normalize(X, norm='l2')
y = mush['class'].values
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3,random_state=123) # 70% training and 30% test

hyperparam={'kernel':['linear', 'poly', 'rbf'], 'gamma' :['scale', 'auto']}
for kernel in hyperparam['kernel']:
    for gamma in hyperparam['gamma']:
        clf = svm.SVC(kernel=kernel,gamma=gamma )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Accuracy for kernel {kernel} and gamma {gamma} : {metrics.accuracy_score(y_test, y_pred)}")
        print(f"Precision for kernel {kernel} and gamma {gamma} : {metrics.precision_score(y_test, y_pred)}")
        print(f"Recall for kernel {kernel} and gamma {gamma} : {metrics.recall_score(y_test, y_pred)}")

# TRY NAIVE BAYES
print("\n-> NAIVE BAYES\n")
X = mush.drop('class', axis=1)
y = mush['class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=123) # 70% training and 30% test
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Accuracy Naive Bayes :",metrics.accuracy_score(y_test, y_pred))

# WITH ONEHOTENCODING

# LabelEncode each column -> caveat, shouldn't I be using OneHotEncoder, LabelEncode should only be for label
#for column in mush.drop('class', axis=1).columns: # I first skipped my label/target but I need to include it too

mush= pd.read_csv("/home/becode/LearnAI/scikit/scikit/Supervised Learning/datasets_478_974_mushrooms.csv")
#veil-type has unique value 'p' so I can drop that column
mush=mush.drop('veil-type', axis=1) # axis=1 means drop column, axis=0 is row
print(mush.head(10))

OH_enc = preprocessing.OneHotEncoder()
x = mush['odor'].values
print(x)
X= np.reshape(x,(-1,1))
print(X)
encoded = OH_enc.fit_transform(X).toarray()
# encoded zonder toarray() geeft iets raars, dus als ik toarray() doe dan krijg ik een array met grootte
# het aantal unieke elementen maal de kolomlengte van de dataframe
print(encoded.size)
print(encoded)

encoder=preprocessing.OneHotEncoder(cols=mush.columns.drop('class',axis=1),handle_unknown='return_nan',return_df=True,use_cat_names=True)
print(encoder)
"""
concat = np.array()
for column in mush.columns:
    #mush[column] = OH_enc.fit_transform(mush[column]).to_array()
    x = mush[column].values
    X= np.reshape(x,(-1,1))
    print(X)
    encoded = OH_enc.fit_transform(X).toarray()
    concat += np.concatenate((concat, encoded), axis=0)
print(concat)
print(concat.size)
print(concat.shape)
print(concat.ndim)
"""

