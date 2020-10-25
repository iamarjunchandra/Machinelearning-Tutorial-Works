# Import dataset
dataSet = read.csv('Position_Salaries.csv')
dataSet = dataSet[, 2:3]

# Fit to SVR regressor
#install.packages('e1071')
library(e1071)
regressor = svm(formula = Salary~.,
                data = dataSet,
                type = 'eps-regression')

# Visualize the result
library(ggplot2)
xGrid = seq(min(dataSet$Level), max(dataSet$Level), .1)
ggplot()+
  geom_point(aes(x = dataSet$Level, y = dataSet$Salary), colour = 'RED')+
  geom_line(aes(x = xGrid, y = predict(regressor,
                                               newdata = data.frame(Level = xGrid))), colour = 'BLUE')+
  ggtitle('Position VS salary')+
  xlab('Level')+
  ylab('Salary')

# Predicting salary of any point
yPred = predict(regressor, newdata = data.frame(Level = 5))