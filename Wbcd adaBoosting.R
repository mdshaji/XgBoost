# Load the Data
# Note: Adaboost can only be run for classification trees. 
# Regression trees cannot be run in R

# wbcd_classification.csv
data = read.csv(file.choose())

##Exploring and preparing the data ----
str(data)

data$diagnosis = as.factor(data$diagnosis)
str(data)

library(caTools)
set.seed(0)
split <- sample.split(data$diagnosis, SplitRatio = 0.7)
train <- subset(data, split == TRUE)
head(train)
test <- subset(data, split == FALSE)

summary(train)

# install.packages("adabag")
library(adabag)

data$diagnosis = as.factor(data$diagnosis)

adaboost <- boosting(diagnosis ~ ., data = train, boos = TRUE,mfinal = 100,
                     coeflearn = "Freund", control = rpart.control(maxdepth = 6) )

# Test data
adaboost_test = predict(adaboost, test)

table(adaboost_test$class, test$diagnosis)

mean(adaboost_test$class == test$diagnosis)


# Train data
adaboost_train = predict(adaboost, train)

table(adaboost_train$class, train$diagnosis)

mean(adaboost_train$class == train$diagnosis)

