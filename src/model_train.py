"""
 This file will build a classifier model for survival prediction using features like sex and passenger class.
 Data: Titanic Dataset
 Algorithm: Decision Tree
 Tools: Pandas, sklearn, joblib
Author: Karthik Kaiplody
"""

import pandas as pd
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from feature.transform import one_hot_encode, train_test_split

# Load the data
dataset = pd.read_csv("./Data/titanic_train.csv")

# Prepare the X and y data
data = dataset[['sex', 'pclass']].copy()
label = dataset['survived']

# Transform the data
data = one_hot_encode(data, 'sex')

# Split the data
data_train, data_test, label_train, label_test = train_test_split(data, label)

# Create a model
clf = DecisionTreeClassifier()
clf.fit(data_train, label_train)
score = clf.score(data_test, label_test)

print(f"Score : {score}")
print(f"Report: \n {classification_report(label_test, clf.predict(data_test))}")

# Store the model
dump(clf, "./Models/clf.bin")



