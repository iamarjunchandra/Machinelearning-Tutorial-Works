#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the Dataset
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

# Rearranging categories
trainData['matchType'].replace(['squad-fpp', 'normal-squad-fpp', 'normal-squad', 'flarefpp', 'flaretpp'], 'squad',inplace=True)
trainData['matchType'].replace(['duo-fpp', 'normal-duo-fpp', 'normal-duo', 'crashfpp', 'crashtpp'], 'duo',inplace=True)
trainData['matchType'].replace(['solo-fpp','normal-solo','normal-solo-fpp'], 'solo',inplace=True)
testData['matchType'].replace(['squad-fpp', 'normal-squad-fpp', 'normal-squad', 'flarefpp', 'flaretpp'], 'squad',inplace=True)
testData['matchType'].replace(['duo-fpp', 'normal-duo-fpp', 'normal-duo', 'crashfpp', 'crashtpp'], 'duo',inplace=True)
testData['matchType'].replace(['solo-fpp','normal-solo','normal-solo-fpp'], 'solo',inplace=True)
trainData['matchType'].value_counts().plot.bar()

# Remove any Missing Values
trainData.dropna(inplace=True)
testData.dropna(inplace=True)
print('NAN Item Row Removed')

# SPlit data as independent and dependent
x = trainData.drop(['winPlacePerc'], axis=1).head(20000).values
y = trainData['winPlacePerc'].head(20000).values

# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
xLinear = LabelEncoder()
xTrain[:, 12] = xLinear.fit_transform(xTrain[:, 12])
hotEnc = OneHotEncoder(categorical_features=[12])
xTrain = hotEnc.fit_transform(xTrain).toarray()
# For Testing Data
xLinear1 = LabelEncoder()
xTest[:, 12] = xLinear1.fit_transform(xTest[:, 12])
hotEnc2 = OneHotEncoder(categorical_features=[12])
xTest = hotEnc2.fit_transform(xTest).toarray()

# Remove Dummy trap
xTrain = xTrain[:, 1:]
xTest = xTest[:, 1:]

# Split data to train and test set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = .3)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
xTrain = scX.fit_transform(xTrain)
xTest = scX.transform(xTest)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(xTrain, yTrain)

from sklearn.model_selection import GridSearchCV
parameters = [{'kernel': ['linear']},
              {'kernel': ['rbf']},
              {'kernel': ['sigmoid']},
              {'kernel': ['poly']}]
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(xTrain, yTrain)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_
1/26