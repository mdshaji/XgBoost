###############Start of code##########
# Importing Dataset
Diabetes.RF = read.csv(file.choose())
str(Diabetes.RF)

#Accuracy with single model
library(caret)
inTraininglocal <- createDataPartition(Diabetes.RF$Class.variable, p = 0.75, list = F)
training <- Diabetes.RF[inTraininglocal, ]
testing <- Diabetes.RF[-inTraininglocal, ]

#install.packages("C50")
library(C50)
model <- C5.0(training$Class.variable ~ ., data = training[, -9])
plot(model)
pred <- predict.C5.0(model, testing[, -9])
a <- table(testing$Class.variable, pred)

sum(diag(a))/sum(a) # 0.671875

#*****************************************************************
########Bagging
acc <- c()
for(i in 1:11)
{
  inTraininglocal <- createDataPartition(Diabetes.RF$Class.variable, p = 0.75, list = F)
  training1 <- Diabetes.RF[inTraininglocal, ]
  testing <- Diabetes.RF[-inTraininglocal, ]
  fittree <- C5.0(training1$Class.variable ~ ., data = training1[, -9])
  pred <- predict.C5.0(fittree,testing[ , -9])
  a <- table(testing$Class.variable, pred)
  acc <- c(acc,sum(diag(a))/sum(a))
}
acc
mean(acc) # 0.735322

#**************************************************************
############## Boosting

# Accuracy with single model with Boosting

inTraininglocal <- createDataPartition(Diabetes.RF$Class.variable, p = 0.75, list = F)
training <- Diabetes.RF[inTraininglocal, ]
testing <- Diabetes.RF[-inTraininglocal, ]
View(inTraininglocal)

model <- C5.0(Diabetes.RF$Class.variable ~ ., data = training[, -9], trials = 10)
pred <- predict.C5.0(model, testing[, -9])
a <- table(testing$Class.variable, pred)

sum(diag(a))/sum(a) # 0.75

#***************************************************************
######### Bagging and Boosting
acc <- c()
for(i in 1:11)
{
  
  inTraininglocal <- createDataPartition(Diabetes.RF$Class.variable, p = 0.75, list = F)
  training1 <- Diabetes.RF[inTraininglocal, ]
  testing <- Diabetes.RF[-inTraininglocal, ]
  
  fittree <- C5.0(training1$Class.variable ~ ., data = training1, trials = 10)
  pred <- predict.C5.0(fittree, testing[, -9])
  a <- table(testing$Class.variable, pred)
  
  acc <- c(acc, sum(diag(a))/sum(a))
  
}

acc
mean(acc) # 0.757
