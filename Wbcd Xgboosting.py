import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/Users/SHAJIUDDIN MOHAMMED/Desktop/wbcd.csv")

df = df.drop(columns = "id")

# Dummy variables
df.head()
df.info()

# Creation of a Dummy Variable for output
lb = LabelEncoder()
df["diagnosis"] = lb.fit_transform(df["diagnosis"])


# Input and Output Split
predictors = df.loc[:, df.columns!="diagnosis"]
type(predictors)

target = df["diagnosis"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)


import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_job = -1)

xgb_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, xgb_clf.predict(x_test))
accuracy_score(y_test, xgb_clf.predict(x_test))
# 0.973

xgb.plot_importance(xgb_clf)

xgb_clf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state =42)

param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0,9],
               'rag_alpha': [1e-2, 0.1, 1]}

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(xgb_clf, param_test1,n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(x_train, y_train)

cv_xg_clf = grid_search.best_estimator_

accuracy_score(y_test, cv_xg_clf.predict(x_test))
#0.96
