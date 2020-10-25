# Multiple Linear Regression (Backward selection model)

#import library
import pandas as pa
import matplotlib.pyplot as plt
import numpy as nm

# import the dataset
dataSet = pa.read_csv('50_Startups.csv')
x = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, 4].values

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label = LabelEncoder()
x[:, 3] = label.fit_transform(x[:, 3])
enc = OneHotEncoder(categorical_features=[3])
x = enc.fit_transform(x).toarray()

# Remove a dummy variable to avoid dummy trap
x = x[:, 1:]

# Split dataset to train set and test set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=.2, random_state = 0)

# Fit the data to linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xTrain, yTrain)

# Predict the Profit for the test set
yPred = regressor.predict(xTest)

# Building the optimum model using backward elimination
import statsmodels.formula.api as sm
# Appending the constant term
x = nm.append(nm.ones((50, 1)).astype(int), values=x, axis=1) 
#Full model with all possible predictors
xOpt = x[:, [0, 1, 2, 3, 4, 5]] 
# Fitting the FUll model
regressorOLS = sm.OLS(endog = y, exog = xOpt).fit() 
# Find the P value of each independent variable
regressorOLS.summary()
""" Since P value of index 2 greater than SL = .05, 
remove the term and fit the model again, repeat the steps until
no pvalue > SL"""
xOpt = xOpt[:, [0, 1, 3, 4, 5]] 
regressorOLS = sm.OLS(endog = y, exog = xOpt).fit() 
regressorOLS.summary()
# Remove Index 1 since P greater than SL
xOpt = xOpt[:, [0, 2, 3, 4]] 
regressorOLS = sm.OLS(endog = y, exog = xOpt).fit() 
regressorOLS.summary()
# Remove Index 2 since P greater than SL
xOpt = xOpt[:, [0, 1, 3]] 
regressorOLS = sm.OLS(endog = y, exog = xOpt).fit() 
regressorOLS.summary()
# Remove Index 2 since P greater than SL
xOpt = xOpt[:, [0, 1]] 
regressorOLS = sm.OLS(endog = y, exog = xOpt).fit() 
regressorOLS.summary()

# Optimum Profit prediction based on R&D Spending
newRegressor = LinearRegression()
newRegressor.fit(xTrain[:, [2]], yTrain)
yOpt = newRegressor.predict(xTest[:, [2]])

# Visualize the data
plt.scatter(xTest[:, [2]], yTest, color = 'RED')
plt.plot(xTest[:, [2]], yOpt, color = 'BLUE')
plt.title('R&D Spending VS Profit (Test Set)')
plt.xlabel('R&D Spending')
plt.ylabel('Profit')
plt.show