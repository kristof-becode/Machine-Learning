from sklearn import preprocessing
import numpy as np

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

print(X_train.mean())
print(X_train.mean(axis=0)) # axis = 0 means over columns
print(X_train.mean(axis=1)) # axis = 1 means over rows

X_scaled = preprocessing.scale(X_train) # standardize around mean = 0
print(X_scaled)
print(X_scaled.mean(axis=0)) # mean over columns gives mean 0 , array([0., 0., 0.])
print(X_scaled.std(axis=0)) # std over columns goves std 1, array([1., 1., 1.])
