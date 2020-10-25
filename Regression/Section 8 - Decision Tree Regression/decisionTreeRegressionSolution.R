# Decision Tree Regression

# Importing the dataset
dataSet = read.csv('Position_Salaries.csv')
dataSet = dataSet[2:3]

# Fitting Decision Tree Regression to the dataset
#install.packages('rpart')
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = dataSet,
                  control = rpart.control(minsplit = 1))

# Predicting a new result with Decision Tree Regression
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the Decision Tree Regression results (higher resolution)
# install.packages('ggplot2')
library(ggplot2)
xGrid = seq(min(dataSet$Level), max(dataSet$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataSet$Level, y = dataSet$Salary),
             colour = 'red') +
  geom_line(aes(x = xGrid, y = predict(regressor, newdata = data.frame(Level = xGrid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression)') +
  xlab('Level') +
  ylab('Salary')

# Plotting the tree
plot(regressor)
text(regressor)