# Load the Data
# Note: Adaboost can only be run for classification trees. 
# Regression trees cannot be run in R

# Importing Dataset
wbcd = read.csv(file.choose())
View(wbcd)

# Removing unwanted columns
wbcd = wbcd[ ,2:32]
View(wbcd)

##Exploring and preparing the data ----
str(wbcd)

library(caTools)
set.seed(0)
split <- sample.split(wbcd$diagnosis, SplitRatio = 0.8)
train <- subset(wbcd, split == TRUE)
test <- subset(wbcd, split == FALSE)

summary(train)
attach(train)
# install.packages("xgboost")
library(xgboost)

train_y <- train$diagnosis == "1"


train_x <- model.matrix(train$diagnosis ~ . -1, data = train)
View(train_x)
# creates dummy variables on attributes
train_x <- train_x[, -1]
View(train_x)
# delete the additional variables     

test_y <- test$diagnosis == "1"

test_x <- model.matrix(test$diagnosis ~ .-1, data = test)
test_x <- test_x[, -1]

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

# Conclusin:
# As per above analysis both test & train accuracy result same i.e., 1
# Hence, the model is best fit
