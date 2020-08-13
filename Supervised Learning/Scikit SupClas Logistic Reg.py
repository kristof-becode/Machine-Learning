import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# import the class for Logistoc Regression
from sklearn.linear_model import LogisticRegression

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

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
"""
Here, you can see the confusion matrix in the form of the array object. 
The dimension of this matrix is 2*2 because this model is binary classification. 
You have two classes 0 and 1. Diagonal values represent accurate predictions, 
while non-diagonal elements are inaccurate predictions.

[[117  13]  117 : TN true negative , 13 : FP false positive
 [ 24  38]]  24 : FN false negative , 38 : TP true positive
 --------------------------------------------------------
=> so ACCURACY = (TP+TN)/(TP+TN+FP+FN) = #correct/total
=> PRECISION =  TP/(TP+FP) = 38/(38+13)= 74.5%
=> RECALL is about 61% =38/(38+24)=TP/(TP+FN) cause I find 38 of 62=38+24 people with actual diabetes
"""
print(cnf_matrix)


# Visualizing Confusion Matrix using Heatmap
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# Confusion Matrix Evaluation Metrics
"""
ALSO SEE ABOVE FOR MORE ABOUT ACCURACY AND PRECISION
Precision: is about being precise, i.e., how accurate your model is. 
In other words, you can say, when a model makes a prediction, how often it is correct.
In your prediction case, when your Logistic Regression model predicted patients are going to suffer from diabetes,
that patients have 76% of the time.
Recall: If there are patients who have diabetes in the test set and your Logistic Regression model
can identify it 58% of the time.
"""
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

"""
ROC Curve
Receiver Operating Characteristic(ROC) curve is a plot of the true positive rate against the false positive rate. 
It shows the tradeoff between sensitivity and specificity.
"""
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()