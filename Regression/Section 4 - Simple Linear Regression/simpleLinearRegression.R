# Import dataset
data = read.csv('Salary_Data.csv')

# Split data into Train and test set
# install.packages('caTools')
library(caTools)
split = sample.split(data, SplitRatio = 2/3)
trainSet = subset(data, split == TRUE)
testSet = subset(data, split == FALSE)

# Fit the train set to linear regression model
regressor = lm(formula = Salary ~ YearsExperience,
               data = trainSet)

# Predict the Salary for test Set
salaryPredicted = predict(regressor, newdata = testSet)

# Visualize the Predicted values
# install.packages('ggplot2')
library(ggplot2)
ggplot()+
  geom_point(aes(x = trainSet$YearsExperience, y = trainSet$Salary), colour = 'RED')+
  geom_line(aes(x = trainSet$YearsExperience, y = predict(regressor,newdata = trainSet)), colour = 'BLUE')+
  ggtitle('Experience VS Salary (Train Set Result)')+
  xlab('Experience')+
  ylab('Salary')

# Train Set Result
library(ggplot2)
ggplot()+
  geom_point(aes(x = testSet$YearsExperience, y = testSet$Salary), colour = 'RED')+
  geom_line(aes(x = trainSet$YearsExperience, y = predict(regressor,newdata = trainSet)), colour = 'BLUE')+
  ggtitle('Experience VS Salary (Test Set Result)')+
  xlab('Experience')+
  ylab('Salary')