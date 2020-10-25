# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataSet = pd.read_csv('Position_Salaries.csv')
X = dataSet.iloc[:, 1:2].values
y = dataSet.iloc[:, 2].values



# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
yPredict = regressor.predict(6.5)

# Visualising the Random Forest Regression results (higher resolution)
xGrid = np.arange(min(X), max(X), 0.01)
xGrid = xGrid.reshape((len(xGrid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(xGrid, regressor.predict(xGrid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
