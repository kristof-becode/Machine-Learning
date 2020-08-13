# Load libraries
import pandas as pd
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


pd.set_option('display.width',400)
pd.set_option('display.max_columns', 40)

# we'll overwrite the column names with similar but shorter names
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
"""
argument header=0 makes so that I can overwrite the first line with col names with my col_names, see above
List of column names to use. If the file contains a header row, 
then you should explicitly pass header=0 to override the column names. 
Duplicates in this list are not allowed
"""
pima = pd.read_csv("/home/becode/LearnAI/scikit/scikit/Supervised Learning/datasets_228_482_diabetes.csv",
                   header=0, names = col_names)

#print(pima.describe())
print(pima.head(10))

#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
#clf =
clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

from sklearn import tree
feat_names = col_names.remove('label')
fig = plt.figure(figsize=(30,20))
tree.plot_tree(clf, feature_names=feat_names, class_names='label', filled=True)
#fig.savefig("/home/becode/LearnAI/scikit/scikit/Supervised Learning/des_tree")
#plt.show()

"""
OPTMIZING DECISION TREE PERFORMANCE

- criterion : optional (default=”gini”) or Choose attribute selection measure: 
This parameter allows us to use the different-different attribute selection measure. 
Supported criteria are “gini” for the Gini index and “entropy” for the information gain.

- splitter : string, optional (default=”best”) or Split Strategy: 
This parameter allows us to choose the split strategy. 
Supported strategies are “best” to choose the best split and “random” to choose the best random split.

- max_depth : int or None, optional (default=None) or Maximum Depth of a Tree: 
The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than 
min_samples_split samples. The higher value of maximum depth causes overfitting, and a lower value causes underfitting.

In Scikit-learn, optimization of decision tree classifier performed by only pre-pruning. 
Maximum depth of the tree can be used as a control variable for pre-pruning. 
In the following the example, you can plot a decision tree on the same data with max_depth=3. 
Other than pre-pruning parameters, You can also try other attribute selection measure such as entropy.
"""
# Let's gro through the criteria referred to above and vary them
criteria={'criterion':['gini', 'entropy'],'split' :['best', 'random'], 'depth':[None, 1,2,3,4,5]}

for criterion in criteria['criterion']:
    for splitter in criteria['split']:
        for depth in criteria['depth']:
            clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=depth)
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(f"Accuracy for criterion {criterion}, splitter {splitter} and max_depth {depth} : {metrics.accuracy_score(y_test, y_pred)}")

