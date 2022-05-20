import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("C:/Users/SHAJIUDDIN MOHAMMED/Desktop/wbcd.csv")
df.columns

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


from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

ada_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test)) # 0.96

accuracy_score(y_train, ada_clf.predict(x_train)) # 1.0
