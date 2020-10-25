# read data
dataSet = dataSet[, 2:3]

#Insufficient data, Hence no need to split as train and test data

# Fit to polynomial Regresion
dataSet$degree2 = dataSet$Level^2
dataSet$degree3 = dataSet$Level^3
dataSet$degree4 = dataSet$Level^4

regressor = lm(formula = Salary ~ Level + degree2 + degree3 + degree4, data = dataSet)
# Visualize Result
library(ggplot2)
xGrid = seq(min(dataSet$Level), max(dataSet$Level), .1)
ggplot()+
  geom_point(aes(x=dataSet$Level, y=dataSet$Salary), colour = 'RED')+
  geom_line(aes(x=xGrid, y=predict(regressor, newdata = data.frame(Level = xGrid,
                                                                   degree2 = xGrid^2,
                                                                   degree3 = xGrid^3,
                                                                   degree4=xGrid^4))), colour = 'BLUE')+
  ggtitle('Employee Level VS Salary')+
  xlab('Level')+
  ylab('Salary')

# Predicitng salary of any level employee
yPredict = predict(regressor, newdata = data.frame(Level = x,
                                                   degree2 = x^2,
                                                   degree3 = x^3,
                                                   degree4 = x^4))