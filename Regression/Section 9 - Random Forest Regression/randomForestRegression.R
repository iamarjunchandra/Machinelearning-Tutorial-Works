# Random Forest Regression

# Importing the dataSet
dataSet = read.csv('Position_Salaries.csv')
dataSet = dataSet[2:3]


# Fitting Random Forest Regression to the dataSet
#install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataSet[-2],
                         y = dataSet$Salary,
                         ntree = 500)

# Predicting a new result with Random Forest Regression
yPred = predict(regressor, data.frame(Level = 6.5))

# Visualising the Random Forest Regression results (higher resolution)
# install.packages('ggplot2')
library(ggplot2)
xGrid = seq(min(dataSet$Level), max(dataSet$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataSet$Level, y = dataSet$Salary),
             colour = 'red') +
  geom_line(aes(x = xGrid, y = predict(regressor, newdata = data.frame(Level = xGrid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Random Forest Regression)') +
  xlab('Level') +
  ylab('Salary')