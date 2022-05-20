#Invoke the below packages
install.packages("xgboost")
library(xgboost)
library(magrittr)
library(dplyr)
library(Matrix)

# To Perform Xgboost all the data types should be numeric

data = read.csv(file.choose())
attach(data)
colnames(data)

View(data)
str(data)
data = data[-1]

data$diagnosis = as.factor(data$diagnosis)
str(data)

#partition of data
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.8,0.2))
train <- data[ind==1,]
test <- data[ind==2,]

#Normalize the data

normFunc <- function(x){(x-mean(x, na.rm = T))/sd(x, na.rm = T)}

data[2:31] <- apply(data[2:31], 2, normFunc)
data$diagnosis = as.factor(data$diagnosis)
data$diagnosis = as.numeric(data$diagnosis)

head(data)
str(data$diagnosis)

# One hot encoding for Factor Variables

trainm <- sparse.model.matrix(diagnosis ~ ., data = train)
head(trainm)

train_label <- train[,"diagnosis"]

train_Matrix <- xgb.DMatrix(data = as.matrix(trainm), label = train_label)

testm <- sparse.model.matrix(diagnosis ~ ., data = test)

test_label <- test[,"diagnosis"]

test_Matrix <- xgb.DMatrix(data = as.matrix(testm), label = test_label)

#parameters

nc <- length(unique(train_label))
xgb_params <- list("objective" = "multi:softmax",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc+1)
watchlist <- list(train = train_Matrix,test = test_Matrix)

#Model Building

bst_model <- xgb.train(params = xgb_params,
                       data = train_Matrix,
                       nrounds = 100,
                       watchlist = watchlist,
                       eta = 0.01,max_depth = 5,subsample = 0.6,gamma = 3)

#Training and Test Error Plot

e <- data.frame(bst_model$evaluation_log)
plot(e$iter, e$train_mlogloss, col ="blue")
lines(e$iter, e$test_mlogloss, col = "red")

#minimum errror

min(e$test_mlogloss)
e[e$test_mlogloss ==  0.413355,]

# 100th iteration has less error

#feature importance

imp <- xgb.importance(colnames(train_Matrix), model = bst_model)
print(imp)
xgb.plot.importance(imp)

#prediction and confusion matrix

pred <- predict(bst_model, newdata = test_Matrix)

confusion_test <- table(x = test$diagnosis, y = pred)

Accuracy_test <- sum(diag(confusion_test))/sum(confusion_test)
Accuracy_test #0.9734

pred_train <- predict(bst_model, newdata = train_Matrix)  

confusion_train <- table(x = train$diagnosis, y = pred_train)

Accuracy_train <- sum(diag(confusion_train))/sum(confusion_train)
Accuracy_train #0.9736

# By tweaking the gamma, eta, maxdepth, subsample the right fit model is achieved.

