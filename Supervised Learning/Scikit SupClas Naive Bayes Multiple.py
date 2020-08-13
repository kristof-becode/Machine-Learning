#Import scikit-learn dataset library
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Load dataset
wine = datasets.load_wine()

print(wine.feature_names) # print the names of the features

print(wine.target_names) # print the label species(class_0, class_1, class_2)

print(wine.data[0:5]) # print the wine data (top 5 records)

print(wine.target) # print the wine labels (0:Class_0, 1:Class_1, 2:Class_3)

print(wine.data.shape) # print data(feature)shape

print(wine.target.shape) # print target(or label)shape

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=109) # 70% training and 30% test

# MAKE MODEL
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
#Create a Gaussian Classifier
gnb = GaussianNB()
#Train the model using the training sets
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)

#EVALUATE
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
