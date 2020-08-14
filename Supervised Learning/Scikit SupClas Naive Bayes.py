
# Assigning features and label variables
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
weather_encoded=le.fit_transform(weather)
print(weather_encoded)


# Converting string labels into numbers
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)
print("Temp:",temp_encoded)
print("Play:",label)

#Combinig weather and temp into single listof tuples
features=list(zip(weather_encoded,temp_encoded)) # Now it is a list of tuples, otherwise cannot work with it or print it,
# zip just returns iterator
print(features)

"""
GENERATING MODEL
Generate a model using naive bayes classifier in the following steps:
1. Create naive bayes classifier
2. Fit the dataset on classifier
3. Perform prediction
"""

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(features,label)

#Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print("Predicted Value for 1 point:", predicted)
predicted= model.predict(features) # 0:Overcast, 2:Mild
print("Predicted Value for all features:", predicted)

# Calculate accuracy for test set of 30%, the dataset size is not enough to build an actual model, this is just playing
#around

# Import train_test_split function
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3) # 70% training and 30% test
model.fit(features,label)
y_pred = model.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
