# Mulitiple Linear Regression 
# import Dataset
dataSet = read.csv('50_Startups.csv')

# Encode Categorical data
dataSet$State = factor(dataSet$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Split train and test set data
library(caTools)
set.seed(123)
split = sample.split(dataSet$Profit, SplitRatio = .8)
trainSet = subset(dataSet, split==TRUE)
testSet = subset(dataSet, split==FALSE)

# Fit in to multi linear Regression
regressor = lm(formula = Profit~., data = trainSet)
profitPredict = predict(regressor, newdata = testSet)

# Fit the optimum model using backward elimination
regressor = lm(formula = Profit~R.D.Spend+Administration+Marketing.Spend+State, data = trainSet)
summary(regressor)
# Pvalue of State > SL = .5, So deleting it
regressor = lm(formula = Profit~R.D.Spend+Administration+Marketing.Spend, data = trainSet)
summary(regressor)
# Pvalue of Administration > SL = .5, So deleting it
regressor = lm(formula = Profit~R.D.Spend+Marketing.Spend, data = trainSet)
summary(regressor)
# Pvalue of Marketing.Spend > SL = .5, So deleting it
regressor = lm(formula = Profit~R.D.Spend, data = trainSet)
summary(regressor)

#R&D's P < SL. Therefore, R&D spending has the highest statistical importance on profit. 
#Visualizing the prediction
library(ggplot2)
ggplot()+
  geom_point(aes(x = testSet$R.D.Spend, y = testSet$Profit), colour='RED')+
  geom_line(aes(x = testSet$R.D.Spend, y = predict(regressor, newdata = testSet)), colour = 'BLUE')+
  ggtitle('R.D Spending VS Profit')+
  xlab('R.D Spending')+
  ylab('Profit')



