# import libraries
import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt

# import data set
dataSet = pd.read_csv('Position_Salaries.csv')
x = dataSet.iloc[:, 1:2].values
y = dataSet.iloc[:, 2].values

# Insufficient data and hence no train test data split
# Fit to linear regression
from sklearn.linear_model import LinearRegression
reg1 = LinearRegression()
reg1.fit(x, y)

# Fit to Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree=4)
xPoly = polyReg.fit_transform(x)
reg2 = LinearRegression()
reg2.fit(xPoly, y)

#Visualizing the Linear regression result
plt.scatter(x, y, color = 'RED')
plt.plot(x, reg1.predict(x), color = 'BLUE')
plt.title('Employee Position VS Salary')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

""" The visualization graph shows huge deviation of the predicted 
salary from actual salary. Hence linear regression model isn't 
useful in this situation. Therefore, Polynomial model is used"""

# Visualize the polynomial result
#Visualizing the Linear regression result
xGrid = nm.arange(min(x), max(x), .1)
xGrid = xGrid.reshape(len(xGrid),1)
plt.scatter(x, y, color = 'RED')
plt.plot(x, reg2.predict(polyReg.fit_transform(x)), color = 'BLUE')
plt.title('Employee Position VS Salary')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""To predict the salry of a employee of any level,
substitute x with the actual level in the following line of code"""
reg2.predict(polyReg.fit_transform(x))