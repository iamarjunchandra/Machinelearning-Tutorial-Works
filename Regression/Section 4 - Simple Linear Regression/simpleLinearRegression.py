# Simple Linear Regression

# import libraries
import pandas as pa
import matplotlib.pyplot as plot
import numpy as nm

dataSet = pa.read_csv('Salary_Data.csv')
x = dataSet.iloc[:, :-1]
y = dataSet.iloc[:, 1]

# Split data into train and test set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state= 0)

# Fitting Simple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xTrain, yTrain)
yPredict = regressor.predict(xTest)

# Visulaizing Predicted result
plot.scatter(xTrain, yTrain, color = 'RED')
plot.plot(xTrain, regressor.predict(xTrain), color = 'Blue')
plot.title('Experience VS Salary(Train Set)')
plot.xlabel('Experience')
plot.ylabel('Salary')
plot.show

plot.scatter(xTest, yTest, color = 'RED')
plot.plot(xTrain, regressor.predict(xTrain), color = 'Blue')
plot.title('Experience VS Salary(Test Set)')
plot.xlabel('Experience')
plot.ylabel('Salary')
plot.show