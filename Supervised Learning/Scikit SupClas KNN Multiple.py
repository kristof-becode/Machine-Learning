from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

wine = datasets.load_wine()


print(wine.feature_names) # print the names of the features

print(wine.target_names) # print the label species(class_0, class_1, class_2)

print(wine.data[0:5]) # print the wine data (top 5 records)

print(wine.target) # print the wine labels (0:Class_0, 1:Class_1, 2:Class_3)

print(wine.data.shape) # print data(feature)shape

print(wine.target.shape) # print target(or label)shape

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3) # 70% training and 30% test

# Generate model for K=5
#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
#Train the model using the training sets
knn.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = knn.predict(X_test)

# Test accuracy of model
# Model Accuracy, how often is the classifier correct?
print("K= 5 : accuracy:",metrics.accuracy_score(y_test, y_pred))

"""
If I input a value for random_state in train_test_split then my test fraction is always sampled the same way from my
data and my accuracy is constant. If I use the above train_test_split then this will vary because the test data will 
always differ slightly
Underneath I tried to look at different K values
"""
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3,random_state=123) # 70% training and 30% test

# TEST MODEL K=5
print("\n-> Accuracy varies a lot related to the test data, compare K=5 for no random state above and with value for random state below for K=5!!")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("\nFor K=5, as just above, only different test fraction but same test size : accuracy:", metrics.accuracy_score(y_test, y_pred))

# TEST MODEL K=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("\nFor K=1 : accuracy:", metrics.accuracy_score(y_test, y_pred))

# TEST MODEL K=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("\nFor K=3 : accuracy:", metrics.accuracy_score(y_test, y_pred))

# TEST MODEL K=6
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("\nFor K=6 : accuracy:", metrics.accuracy_score(y_test, y_pred))

# TEST MODELK=9
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("\nFor K=9 : accuracy:", metrics.accuracy_score(y_test, y_pred))


"""
For KNN Normalization can make a big difference so let's try it!
"""
#Import preprocessing
from sklearn import preprocessing
X_normalized = preprocessing.normalize(wine.data, norm='l2')
X_train, X_test, y_train, y_test = train_test_split(X_normalized, wine.target, test_size=0.3,random_state=123) # 70% training and 30% test

# TEST MODEL K=5
print("\n-> Accuracy should improve with normalization of dataset, let's see: ")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("\nFor K=5 with normalization : accuracy:", metrics.accuracy_score(y_test, y_pred))

# TEST MODEL K=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("\nFor K=1 with normalization : accuracy:", metrics.accuracy_score(y_test, y_pred))

# TEST MODEL K=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("\nFor K=3 with normalization : accuracy:", metrics.accuracy_score(y_test, y_pred))

# TEST MODEL K=6
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("\nFor K=6 with normalization : accuracy:", metrics.accuracy_score(y_test, y_pred))

# TEST MODELK=9
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("\nFor K=9 with normalization : accuracy:", metrics.accuracy_score(y_test, y_pred))

print("INDEED!! ACCURACY HIGHER OVERALL WITH NORMALIZATION!!!!")


