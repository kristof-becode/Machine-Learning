from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

pd.set_option('display.width',400)
pd.set_option('display.max_columns', 40)

"""
DIABETES DATASET
"Ten baseline variables, age, sex, body mass index, average blood pressure,
and six blood serum measurements were obtained for each of n = 442 diabetes patients, 
as well as the response of interest,
a quantitative measure of disease progression one year after baseline." 
Note that the 10 x variables have been standardized to have mean 0 and squared length = 1 (sum(x^2)=1). (L2 Norm)
The 11th column is target, a diabetes disease marker
"""
data = datasets.load_diabetes()
print (data.DESCR)

df = pd.DataFrame(data.data, columns=data.feature_names) #features
target = pd.DataFrame(data.target)

print(df.head(10))
print(df.describe())
print(target.head(10))

"""
CALCULATE LIN REGRESSION FOR INDIVIDUAL FEATURES
"""
print("\nCalculating relationships diabetes disease marker per column/feature")
relation = []
rsquared= []
intercept = []
slope = []
for item in df.columns:
    x = df[item].values
    X = np.reshape(x,(-1, 1))
    y = target.values
    model = LinearRegression().fit(X, y)
    rel_text = str(item) + " vs disease marker"
    relation.append(rel_text)
    rsquared.append(model.score(X, y))
    intercept.append(model.intercept_)
    slope.append(model.coef_)
    # PLOT
    p = df.columns.tolist().index(item) + 1
    plt.figure(1, figsize=(15, 15))
    plt.subplot(2, 5, p)  # 2 rows, 5 columns, 1st subplot = top left
    plt.scatter(X, y, s=0.75)
    plt.ylabel("disease_marker")
    plt.xlabel(item)
path = "/home/becode/LearnAI/scikit/Diabetes_plots.png"
#plt.savefig(path, transparent=False)
#plt.show()
overview = pd.DataFrame({'Relation':relation, 'R_squared': rsquared, 'Intercept': intercept, 'Slope':slope})
print(overview.sort_values(by='R_squared',ascending=False))

"""
CALCULATE LIN REGRESSION FOR X= ALL 10 FEATURES
"""
print("\nCalculating relationships diabetes disease marker and total features or all 10 columns")
X = df # whole features dataframe with 10 colums or features
y = target.values
model = LinearRegression().fit(X, y)
print("R squared is higher when looking at total features: ", model.score(X, y))

"""
CALCULATE LIN REGRESSION FOR X= SELECTION OF FEATURES
"""
print("\nCalculating relationships diabetes disease marker and selections of features, bmi and s5")
X = df[['bmi','s5']]
model = LinearRegression().fit(X, y)
print("R squared for disease_marker vs bmi and s5: ", model.score(X, y))

print("\nCalculating relationships diabetes disease marker and selections of features")
X = df[['bmi','bp','s5']]
model = LinearRegression().fit(X, y)
print("R squared for disease_marker vs bmi, s5 and bp: ", model.score(X, y))






"""
#BELOW I TRIED TO FIND RELATIONSHIPS BETWEEN THE 10 FEATURES IN THE DATASET AS I DIDN'T REALIZE THAT THERE WERE VALUES IN
 #THE 11th COLUMN? ACCESIBLE VIA TARGET..SO THIS BELOW WAS QUITE A WASTE OF TIME
 
#sns.pairplot(df) # pairplot is busy and heavy, linear relationships are noticeable
#plt.savefig("/home/becode/LearnAI/scikit/Diabetes.png", transparent=False)

#!!!!! PLOTTING
for item in df.columns:
    #print(item)
    x = df[item]
    listY = df.columns.tolist() # list of columns, instead of series df.columns()
    listY.remove(item) # delete the item from previous list
    for y in listY: # let's create a subplot per column to investigate the relationship with other the other columns
        p = listY.index(y) + 1
        plt.figure(1,figsize=(15,15))
        #plt.title("Plotting for ")
        plt.subplot(3, 3, p)  # 3 rows, 3 columns, 1st subplot = top left
        plt.scatter(x, df[y], s=0.75)
        #sns.lmplot(x, df[y], height=1, fit_reg=True)
        #plt.xlabel(item)
        plt.ylabel(y)
    path = "/home/becode/LearnAI/scikit/Diabetes_" + item + "plots.png"
    #plt.savefig(path, transparent=False)
    plt.show()

#"!!!  Rsquared
relation = []
rsquared= []
for item in df.columns:
    print(item)
    x = df[item].values
    X = np.reshape(x,(-1, 1))
    listY = df.columns.tolist() # list of columns, instead of series df.columns()
    listY.remove(item) # delete the item from previous list
    for Y in listY: #
        #relation.append(item)
        y = df[Y].values
        model = LinearRegression().fit(X,y)
        print("\nfor " + item + " vs " + Y + "\n rsquared = " + str(model.score(X, y)) + "\n intercept = " + str(model.intercept_) + "\n slope = " + str(
           model.coef_))
        rel_text = str(item) + " vs " + str(Y)
        relation.append(rel_text)
        rsquared.append(model.score(X, y))
overview = pd.DataFrame({'Relation':relation, 'R_squared': rsquared})
print(overview.sort_values(by='R_squared',ascending=False))
"""