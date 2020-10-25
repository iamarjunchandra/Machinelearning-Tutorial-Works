# Support Vector Regression (SVR)
# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import the dataSet 
dataSet = pd.read_csv('Position_Salaries.csv')
x = dataSet.iloc[:, 1:2].values
y = dataSet.iloc[:, 2:3].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
scY = StandardScaler()
x = scX.fit_transform(x)
y = scY.fit_transform(y)

# Fit the data to regressor (SVR)
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x, y)

# Visualize the result
xGrid = np.arange(min(x), max(x), .1)
xGrid = xGrid.reshape(len(xGrid), 1)
plt.scatter(x, y, color = 'RED')
plt.plot(x, scY.inverse_transform(regressor.predict(x)), color = 'BLUE' )
plt.title('SVR Level VS Salary Prediction')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
#Predict the Salary for any data by replacing X in following line of code
yPredict = scY.inverse_transform(regressor.predict(scX.transform(X)))