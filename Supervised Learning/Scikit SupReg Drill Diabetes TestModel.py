from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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

print("\n->Calculating test model specs diabetes disease marker for individual features")
print("with test data taken from the end of the total X_data array: \n")
feature = []
fraction = []
r_squared = []
mean_sq_err = []
slope = []
for item in df.columns:
    x_data = df[item].values
    X_data = np.reshape(x_data,(-1, 1))
    y_data = target.values
    #print(X_data)
    #print(y_data)
    #print(X_data.size)
    #print(y_data.size)
    # Here I will just take a slice from the end of the dataset to be used as test,
    # would it be an idea to select data at random out of the dataset for testing?
    fraction_list = [0.5, 0.2, 0.1, 0.05]
    for test_fraction in fraction_list:
        #test_fraction = 0.2
        size_X_test = int(X_data.size * (test_fraction))
        size_y_test = int(y_data.size * (test_fraction))
        X_train = X_data[:-size_X_test]
        y_train = y_data[:-size_y_test]
        X_test = X_data[-size_X_test:]
        y_test = y_data[-size_y_test:]

        regr = LinearRegression().fit(X_train,y_train)

        y_pred = regr.predict(X_test)
        feature.append(item)
        fraction.append(test_fraction)
        mean_sq_err.append(mean_squared_error(y_test, y_pred))
        r_squared.append(r2_score(y_test, y_pred))
        slope.append(regr.coef_)
        #print(regr.coef_)
        #print(mean_squared_error(y_test, y_pred))
        #print(r2_score(y_test, y_pred))
overview = pd.DataFrame({'Feature':feature, 'Test Fraction': fraction, 'R_squared': r_squared, 'Mean_squared_error': mean_sq_err, 'Slope':slope})
#overview_sort = overview.sort_values(by='R_squared',ascending=False)
#ov = overview.groupby('Feature')
print(overview)
#overview_group = overview.groupby(by=feature)
#print(overview_group.sort_values(by='Mean_squared_error',ascending=False))
#print(overview.sort_values(by='R_squared',ascending=False))
#print(overview.sort_values(by='Mean_squared_error',ascending=False))

# REPEAT ABOVE(all individual features) WITH TRAIN_TEST_SPLIT TO ADD RANDOMNESS TO SPLITTING TEST/TRAINING DATA
# do remember to keep random_state equal so that you always have the 'same' random split!!!!
print("\n->Calculating test model specs diabetes disease marker for individual features")
print("with train_test_split to random split training and test data: \n")
feature = []
fraction = []
r_squared = []
mean_sq_err = []
slope = []
for item in df.columns:
    x_data = df[item].values
    X_data = np.reshape(x_data,(-1, 1))
    y_data = target.values

    fraction_list = [0.5, 0.2, 0.1, 0.05]
    for test_size in fraction_list:

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=123)

        regr = LinearRegression().fit(X_train,y_train)

        y_pred = regr.predict(X_test)
        feature.append(item)
        fraction.append(test_size)
        mean_sq_err.append(mean_squared_error(y_test, y_pred))
        r_squared.append(r2_score(y_test, y_pred))
        slope.append(regr.coef_)
        #print(regr.coef_)
        #print(mean_squared_error(y_test, y_pred))
        #print(r2_score(y_test, y_pred))
overview = pd.DataFrame({'Feature':feature, 'Test Fraction': fraction, 'R_squared': r_squared, 'Mean_squared_error': mean_sq_err, 'Slope':slope})
#overview_sort = overview.sort_values(by='R_squared',ascending=False)
#ov = overview.groupby('Feature')
print(overview)
ov = overview.groupby('Feature')['Mean_squared_error'].max()
print(ov)
#overview_group = overview.groupby(by=feature)
#print(overview_group.sort_values(by='Mean_squared_error',ascending=False))
#print(overview.sort_values(by='R_squared',ascending=False))
#print(overview.sort_values(by='Mean_squared_error',ascending=False))


# for all ten features
print("\n->Calculating test model specs diabetes disease marker and all 10 features:\n")
fraction = []
r_squared = []
mean_sq_err = []
slope = []

X_data = df # whole features dataframe with 10 columns or features
y_data = target.values

fraction_list = [0.5, 0.2, 0.1, 0.05]
for test_size in fraction_list:
    # test_fraction = 0.2
    #size_X_test = int(X_data.size * (test_fraction))
    #size_y_test = int(y_data.size * (test_fraction))
    #X_train = X_data[:-size_X_test]
    #y_train = y_data[:-size_y_test]
    #X_test = X_data[-size_X_test:]
    #y_test = y_data[-size_y_test:]

    # I HAD TO PROCEED WITH TEST_SPLIT, OTHERWISE I KEPT GETIING ERRORS WHEN TRYING TO CUT THE ARRAY/COLUMN FROM THE END
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=123)
    regr = LinearRegression().fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    fraction.append(test_size)
    mean_sq_err.append(mean_squared_error(y_test, y_pred))
    r_squared.append(r2_score(y_test, y_pred))
    slope.append(regr.coef_)
overview = pd.DataFrame({'Test Fraction': fraction, 'R_squared': r_squared, 'Mean_squared_error': mean_sq_err, 'Slope':slope})
print(overview)


# for a selection of 'bmi','s5'
print("\n->Calculating test model specs diabetes disease marker and selections of features: bmi, s5\n")
fraction = []
r_squared = []
mean_sq_err = []
slope = []
X_data = df[['bmi','s5']]
y_data = target.values
fraction_list = [0.5, 0.2, 0.1, 0.05]
for test_size in fraction_list:
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=123)
    regr = LinearRegression().fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    fraction.append(test_size)
    mean_sq_err.append(mean_squared_error(y_test, y_pred))
    r_squared.append(r2_score(y_test, y_pred))
    slope.append(regr.coef_)
overview = pd.DataFrame({'Test Fraction': fraction, 'R_squared': r_squared, 'Mean_squared_error': mean_sq_err, 'Slope':slope})
print(overview)


# for a selection of 'bmi','s5','bp'
print("\n->Calculating test model specs diabetes disease marker and selections of features: bmi, bp, s5 \n")
fraction = []
r_squared = []
mean_sq_err = []
slope = []

X_data = df[['bmi','s5','bp']]
y_data = target.values

fraction_list = [0.5, 0.2, 0.1, 0.05]
for test_size in fraction_list:
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=123)
    regr = LinearRegression().fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    fraction.append(test_size)
    mean_sq_err.append(mean_squared_error(y_test, y_pred))
    r_squared.append(r2_score(y_test, y_pred))
    slope.append(regr.coef_)
overview = pd.DataFrame({'Test Fraction': fraction, 'R_squared': r_squared, 'Mean_squared_error': mean_sq_err, 'Slope':slope})
print(overview)
