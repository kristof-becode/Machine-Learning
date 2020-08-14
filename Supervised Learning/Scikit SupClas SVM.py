#Import scikit-learn dataset library
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
#Import svm model
from sklearn import svm

#Load dataset
cancer = datasets.load_breast_cancer()

# print the names of the 13 features
print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)

# print data(feature)shape
print("feature shape: ", cancer.data.shape)

# print the cancer data features (top 5 records)
print(cancer.data[0:5])

# print the cancer labels (0:malignant, 1:benign)
print(cancer.target)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision:
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall:
print("Recall:",metrics.recall_score(y_test, y_pred))

"""
TUNING HYPERPARAMETERS

- Kernel: kernel:{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
The main function of the kernel is to transform the given dataset input data into the required form. 
There are various types of functions such as linear, polynomial, and radial basis function (RBF). 
Polynomial and RBF are useful for non-linear hyperplane. Polynomial and RBF kernels compute the separation 
line in the higher dimension. In some of the applications, it is suggested to use a more complex kernel to 
separate the classes that are curved or nonlinear. This transformation can lead to more accurate classifiers.

- Regularization: C:float, default=1.0
Regularization parameter in python's Scikit-learn C parameter used to maintain regularization. 
Here C is the penalty parameter, which represents misclassification or error term.
The misclassification or error term tells the SVM optimization how much error is bearable. 
This is how you can control the trade-off between decision boundary and misclassification term. 
A smaller value of C creates a small-margin hyperplane and a larger value of C creates a larger-margin hyperplane.

- Gamma: gamma:{‘scale’, ‘auto’} or float, default=’scale’
A lower value of Gamma will loosely fit the training dataset, 
whereas a higher value of gamma will exactly fit the training dataset, which causes over-fitting. 
In other words, you can say a low value of gamma considers only nearby points in calculating the separation line, 
while the a value of gamma considers all the data points in the calculation of the separation line.
"""
# Let's gro through the hyperparameters referred to above and vary them
hyperparam={'kernel':['linear', 'poly', 'rbf'], 'gamma' :['scale', 'auto']}
for kernel in hyperparam['kernel']:
    for gamma in hyperparam['gamma']:
        clf = svm.SVC(kernel=kernel,gamma=gamma )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Accuracy for kernel {kernel} and gamma {gamma} : {metrics.accuracy_score(y_test, y_pred)}")
        print(f"Precision for kernel {kernel} and gamma {gamma} : {metrics.precision_score(y_test, y_pred)}")
        print(f"Recall for kernel {kernel} and gamma {gamma} : {metrics.recall_score(y_test, y_pred)}")