# Load the Data
# Note: Adaboost can only be run for classification trees. 
# Regression trees cannot be run in R

# Importing Dataset
Diabetes.RF = read.csv(file.choose())

##Exploring and preparing the data ----
str(Diabetes.RF)

library(caTools)
set.seed(0)
split <- sample.split(Diabetes.RF$Class.variable, SplitRatio = 0.8)
train <- subset(Diabetes.RF, split == TRUE)
test <- subset(Diabetes.RF, split == FALSE)

summary(train)

# install.packages("adabag")
library(adabag)

train$Class.variable <- as.factor(train$Class.variable)

adaboost <- boosting(Class.variable ~ ., data = train, boos = TRUE)

# Test data
adaboost_test = predict(adaboost, test)
table(adaboost_test$class, test$Class.variable)
mean(adaboost_test$class == test$Class.variable)
# Test Accuracy = 0.727

# Train data
adaboost_train = predict(adaboost, train)
table(adaboost_train$class, train$Class.variable)
mean(adaboost_train$class == train$Class.variable)
# Train Accuracy = 1

# The model clearly shows as overfit,need to do regularization to make the model as right fit

###################################################################################################


# Load the Data
# Note: XGboosting data should be convereted into Dmatrix 

# Importing Dataset
Diabetes.RF = read.csv(file.choose())

##Exploring and preparing the data ----
str(Diabetes.RF)

library(caTools)
set.seed(0)
split <- sample.split(Diabetes.RF$Class.variable, SplitRatio = 0.8)
train <- subset(Diabetes.RF, split == TRUE)
test <- subset(Diabetes.RF, split == FALSE)

summary(train)
attach(train)
# install.packages("xgboost")
library(xgboost)

train_y <- train$Class.variable == "1"


train_x <- model.matrix(train$Class.variable ~ . -1, data = train)
# creates dummy variables on attributes
train_x <- train_x[, -9]
View(train_x)
# delete the additional variables     

test_y <- test$Class.variable == "1"

test_x <- model.matrix(test$Class.variable ~ .-1, data = test)
test_x <- test_x[, -9]

Xmatrix_train <- xgb.DMatrix(data = train_x, label = train_y)
Xmatrix_test <- xgb.DMatrix(data = test_x, label = test_y)


xg_boosting <- xgboost(data = Xmatrix_train, nround = 50,
                       objective = "multi:softmax", eta = 0.3, 
                       num_class = 2, max_depth = 100)


# Prediction for test data
xgbpred_test <- predict(xg_boosting, Xmatrix_test)

table(test_y, xgbpred_test)
mean(test_y == xgbpred_test)

# Prediction for train data
xgbpred_train <- predict(xg_boosting, Xmatrix_train)

table(train_y, xgbpred_train)
mean(train_y == xgbpred_train)

