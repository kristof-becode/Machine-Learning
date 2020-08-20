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
mush= pd.read_csv("/home/becode/LearnAI/scikit/scikit/Supervised Learning/datasets_478_974_mushrooms.csv")

print(mush.head(10))
# Observation : class = the label/target with values p(oisonous) and e(dible),
# the other 22 columns are possible features
# Observation : all the data is discrete and categorical, nominal but not ordinal
print(f"\nSome stats:\n ",mush.describe())
# Observation : all the columns are discrete values => take into account for Classifier selection

print(f"\nshape dataframe: {mush.shape}")
print(f"\n{mush[mush['class'] == 'p']['class'].count()} rows 'poisonous' on total {mush.shape[0]} rows -> "
      f" {round(((mush[mush['class'] == 'p']['class'].count()) / (mush.shape[0])) *100,2)}%")
print(f"{mush[mush['class'] == 'e']['class'].count()} rows 'edible' on total {mush.shape[0]} rows -> "
      f" {round(((mush[mush['class'] == 'e']['class'].count()) / (mush.shape[0])) *100,2)}%")
print("\nChecking for NA values:\n",mush.isna().sum()) # check for NA values in dataset
# Observation: No NA values in dataset

# Observation: 'veil-type' column has only one unique value 'p' so I can drop that column
mush=mush.drop('veil-type', axis=1) # axis 1 means drop column, axis= 0 is row


"""
LABEL ENCODING THE DATA
    LabelEncode each column -> caveat, shouldn't I be using OneHotEncoder? => (Later, below)!
"""

# Call Label Encoder class and encode all columns
lab_enc = preprocessing.LabelEncoder()
for column in mush.columns:
    mush[column] = lab_enc.fit_transform(mush[column])
# !!!!!!! (p)oisonous = 1, (e)dible = 0 !!!!!!
print(mush.head(10))

# Check correlation matrix
print("\n->Correlation matrix for Pearson coeff:")
corr_matrix = mush.corr()
print(corr_matrix["class"].sort_values(ascending=False))


# check which columns or features to drop
# Test Lin Reg for individual features
print("-> Linear Regression on Label Encoded columns or possible features:")
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
# X = np.reshape(x, (-1, 1))
y = mush['class'].values
model = LinearRegression().fit(X, y)
print(f"R squared for class vs 'gill-size','gill-color','bruises','ring-type':{model.score(X, y)}")

# Test Lin Reg for selection of features
X = mush[['gill-size','gill-color','bruises','ring-type','stalk-root']]
# X = np.reshape(x, (-1, 1))
y = mush['class'].values
model = LinearRegression().fit(X, y)
print(f"R squared for class vs 'gill-size','gill-color','bruises','ring-type','stalk-root':{model.score(X, y)}")

# Test Lin Reg for all features combined - THIS SCORES THE BEST REGRESSION
X = mush.drop('class', axis=1)
# X = np.reshape(x, (-1, 1))
y = mush['class'].values
model = LinearRegression().fit(X, y)
print(f"R squared for class vs all columns:{model.score(X, y)}")


# TRY KNN for different K-VALUES AND NORMALIZED DATASET
print("\n-> CLF: KNN\n")
X = mush.drop('class', axis=1)
X_normalized = preprocessing.normalize(X, norm='l2')
y = mush['class'].values
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3,random_state=123) # 70% training and 30% test
k_list = [1,3,5,9]
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"KNN for K={k} with normalization; accuracy: {metrics.accuracy_score(y_test, y_pred)} | "
          f"precision: {metrics.precision_score(y_test, y_pred)} | recall: {metrics.recall_score(y_test, y_pred)} ")

# TRY LOGISTIC REGRESSION
print("\n-> CLF: LOGISTIC REGRESSION\n")
X = mush.drop('class', axis=1)
stan = preprocessing.StandardScaler()
X_standardized = stan.fit_transform(X)
# X_standardized = preprocessing.StandardScaler(X)
y = mush['class'].values
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3,random_state=123) # 70% training and 30% test
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print(f"\nLogistic regression; accuracy: {metrics.accuracy_score(y_test, y_pred)} | "
      f"precision: {metrics.precision_score(y_test, y_pred)} | recall: {metrics.recall_score(y_test, y_pred)}")

# TRY DECISION TREE, NO SCALING NEEDED
print("\n-> CLF: DECISION TREE\n")
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
            print(f"Dec Tree for criterion {criterion}, splitter {splitter} and max_depth {depth}; "
                  f"accuracy: {metrics.accuracy_score(y_test, y_pred)} |"
                  f" precision: {metrics.precision_score(y_test, y_pred)} |"
                  f" recall: {metrics.recall_score(y_test, y_pred)}")


# TRY SVM, NORMALIZATION GOOD IDEA
print("\n-> CLF: SVM\n")
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
        print(f"SVM for kernel {kernel} and gamma {gamma}; accuracy: {metrics.accuracy_score(y_test, y_pred)} | "
              f"precision: {metrics.precision_score(y_test, y_pred)} | recall: {metrics.recall_score(y_test, y_pred)}")


# TRY NAIVE BAYES
print("\n-> CLF: NAIVE BAYES\n")
X = mush.drop('class', axis=1)
y = mush['class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=123) # 70% training and 30% test
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(f"Naive Bayes; accuracy: {metrics.accuracy_score(y_test, y_pred)} |"
      f" precision: {metrics.precision_score(y_test, y_pred)} | recall: {metrics.recall_score(y_test, y_pred)}")



"""
ONE HOT ENCODING THE DATA
    Ideal for discrete, nominal categorical data
"""

mush= pd.read_csv("/home/becode/LearnAI/scikit/scikit/Supervised Learning/datasets_478_974_mushrooms.csv")

# BAR PLOTS !!!!!!!!!!

fig1 = plt.figure(figsize=(200,120))
for column in mush.drop('class',axis=1).columns:
    p = mush[mush['class'] == 'p'][column].value_counts()
    e = mush[mush['class'] == 'e'][column].value_counts()
    elements = mush[column].unique()
    pois = []
    edible = []
    for el in elements:
        if el in p:
            pois.append(p.loc[el])
        else:
            pois.append(0)
        if el in e:
            edible.append(e.loc[el])
        else:
            edible.append(0)
    p = mush.drop('class',axis=1).columns.tolist().index(column) + 1
    ax = fig1.add_subplot(4, 6, p)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.bar(elements, pois, color='red', edgecolor='white', label='poisonous')
    plt.bar(elements, edible, bottom=pois, color='blue', edgecolor='white',label='edible')
    #plt.legend()
    plt.title(column)
plt.legend(loc='lower right')
plt.show()


# ONE HOT FOR REAL
mush_without = mush.drop('class',axis=1)
OH = preprocessing.OneHotEncoder()
lab_enc = preprocessing.LabelEncoder()
mush_OH_enc = OH.fit_transform(mush_without)
class_lab_enc = lab_enc.fit_transform(mush['class'])
print(mush_OH_enc.toarray())
print(class_lab_enc)

"""
# Check correlation matrix
kopij = mush
for column in kopij.columns:
    kopij[column] = OH.fit_transform(kopij[column])
corr_matrix = kopij.corr()
print(corr_matrix["class"].sort_values(ascending=False))
"""
# CHECK CHI2: feature selection for OneHotEncoded DataFrame
X=mush_without
y=class_lab_enc
fs = SelectKBest(score_func=chi2, k='all')
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3,random_state=123) # 70% training and 30% test
fs.fit(X_train, y_train)
X_train_fs = fs.transform(X_train)
X_test_fs = fs.transform(X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()


# Test Lin Reg for individual features
relation = []
rsquared = []
for column in mush.drop('class',axis=1).columns:
    X = mush_OH_enc.toarray()
    #X = np.reshape(x,(-1, 1))
    y = class_lab_enc
    model = LinearRegression().fit(X, y)
    relation.append(f"Class vs {column}")
    rsquared.append(model.score(X, y))
    #print(f"R squared for class vs {column}:{model.score(X, y)}")
r_scores = pd.DataFrame({'Relation': relation, 'R²':rsquared})
print(r_scores.sort_values(by='R²', ascending=False))


# TRY KNN for different K-VALUES AND NORMALIZED DATASET
print("\n-> CLF: KNN\n")
X = mush_OH_enc
X_normalized = preprocessing.normalize(X, norm='l2')
y = class_lab_enc
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3,random_state=123) # 70% training and 30% test
k_list = [1,3,5,9]
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"KNN for K={k} with normalization; accuracy : {metrics.accuracy_score(y_test, y_pred)}| "
          f"precision : {metrics.precision_score(y_test, y_pred)}| recall : {metrics.recall_score(y_test, y_pred)} ")

# TRY LOGISTIC REGRESSION
print("\n-> CLF: LOGISTIC REGRESSION\n")
X = mush_OH_enc.toarray()
stan = preprocessing.StandardScaler()
X_standardized = stan.fit_transform(X)
#X_standardized = preprocessing.StandardScaler(X)
y = class_lab_enc
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3,random_state=123) # 70% training and 30% test
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print(f"\nLogistic regression; accuracy: {metrics.accuracy_score(y_test, y_pred)} | "
      f"precision : {metrics.precision_score(y_test, y_pred)}| recall : {metrics.recall_score(y_test, y_pred)}")

# TRY DECISION TREE , NO SCALING NEEDED
print("\n-> CLF: DECISION TREE\n")
X = mush_OH_enc
y = class_lab_enc
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
            print(f"Dec Tree for criterion {criterion}, splitter {splitter} and max_depth {depth}; "
                  f"accuracy: {metrics.accuracy_score(y_test, y_pred)} |"
                  f" precision : {metrics.precision_score(y_test, y_pred)} |"
                  f" recall : {metrics.recall_score(y_test, y_pred)}")

# TRY SVM # NORMALIZATION GOOD IDEA
print("\n-> CLF: SVM\n")
X = mush_OH_enc.toarray()
X_normalized = preprocessing.normalize(X, norm='l2')
y = class_lab_enc
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=123) # 70% training and 30% test

hyperparam={'kernel':['linear', 'poly', 'rbf'], 'gamma' :['scale', 'auto']}
for kernel in hyperparam['kernel']:
    for gamma in hyperparam['gamma']:
        clf = svm.SVC(kernel=kernel,gamma=gamma )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"SVM for kernel {kernel} and gamma {gamma}; accuracy: {metrics.accuracy_score(y_test, y_pred)} | "
              f"precision: {metrics.precision_score(y_test, y_pred)} | recall: {metrics.recall_score(y_test, y_pred)}")

# TRY NAIVE BAYES
print("\n-> CLF: NAIVE BAYES\n")
X = mush_OH_enc.toarray()
y = class_lab_enc
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=123) # 70% training and 30% test
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(f"Naive Bayes; accuracy: {metrics.accuracy_score(y_test, y_pred)} |"
      f" precision: {metrics.precision_score(y_test, y_pred)} | recall: {metrics.recall_score(y_test, y_pred)}")

"""
TRYING TO COMBINE ALL CLF IN ONE FUNCTION - NOT FINISHED
"""
#clf_data = {'clf_type':['KNN','LogReg','SVM','DecTree','Naive Bayes'] , 'class':['KNeighborsClassifier(n_neighbors=k)','LogisticRegression()' ,'DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=depth)','svm.SVC(kernel=kernel,gamma=gamma)','GaussianNB()'],'Norm_Stan':[[1,0],[0,1],[1,0],[0,0],[0,0]], 'params':[{'n_neighbors':[1,2,3,4,5]},{'criterion':['gini', 'entropy'],'split' :['best', 'random'], 'depth':[None, 1,2,3,4,5]},{'kernel':['linear', 'poly', 'rbf'], 'gamma' :['scale', 'auto']}, {}]}

#X =
#y =

#i=0

"""    
if clf_data[Norm_Stan][0] == 1:
    X = preprocessing.normalize(X, norm='l2')
if clf_data[Norm_Stan][1] == 1:
    stan = preprocessing.StandardScaler()
    X = stan.fit_transform(X)
"""