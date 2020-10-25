# dECISION tREE rEGRESSION

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataSet = pd.read_csv('Position_Salaries.csv')
X = dataSet.iloc[:, 1:2].values
y = dataSet.iloc[:, 2].values

# Fitting the Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
yPredict = regressor.predict(6.5)

# Visualising the Regression results
XGrid = np.arange(min(X), max(X), 0.01)
XGrid = XGrid.reshape((len(XGrid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(XGrid, regressor.predict(XGrid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
