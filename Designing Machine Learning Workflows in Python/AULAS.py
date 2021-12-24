# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
#Algebra Linear
import numpy as np 

# Processamento/manipulação dos dados
import pandas as pd 

# Visualização dos dados
import seaborn as sns
# %matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style

#Algoritmos Machine Learning
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
# -

credit = pd.read_csv("credit.csv")
#print(credit.dtypes)

# +

from sklearn.preprocessing import LabelEncoder
# -

print(credit.info())
non_numeric_columns = credit.select_dtypes(include=['object']).copy()

# +
# Inspect the first few lines of your data using head()
credit.head(3)

# Create a label encoder for each column. Encode the values
for column in non_numeric_columns:
    le = LabelEncoder()
    credit[column] = le.fit_transform(credit[column])

# Inspect the data types of the columns of the data frame
print(credit.dtypes)

# +
#print(non_numeric_columns)

# +
#Tutorial muito bom para preparar os dados https://www.datacamp.com/community/tutorials/categorical-data
# -

credit.head(5)
credit.describe()
X = credit.drop("class", axis=1)
y = credit["class"]
#X_test  = test_df.drop("PassengerId", axis=1).copy()
from sklearn.model_selection import train_test_split

# +

# Split the data into train and test, with 20% as test
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=1)
# -

from sklearn.model_selection import learning_curve, GridSearchCV

# +

# Set a range for n_estimators from 10 to 40 in steps of 10
param_grid = {'n_estimators': range(10, 50, 10)}

# Optimize for a RandomForestClassifier() using GridSearchCV
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid.fit(X, y)
grid.best_params_

# +
# Create numeric encoding for credit_history
credit_history_num = LabelEncoder().fit_transform(
  credit['credit_history'])

# Create a new feature matrix including the numeric encoding
X_num = pd.concat([X, pd.Series(credit_history_num)], 1)

# Create new feature matrix with dummies for credit_history
X_hot = pd.concat(
  [X, pd.get_dummies(credit['credit_history'])], 1)

# Compare the number of features of the resulting DataFrames
X_hot.shape[1] > X_num.shape[1]

# +
from sklearn.feature_selection import SelectKBest, f_classif, chi2

# Function computing absolute difference from column mean
def abs_diff(x):
    return np.abs(x-np.mean(x))

# Apply it to the credit amount and store to new column
credit['diff'] = abs_diff(credit['credit_amount'])

# Create a feature selector with chi2 that picks one feature
sk = SelectKBest(chi2, k=1)

# Use the selector to pick between credit_amount and diff
sk.fit(credit[['credit_amount', 'diff']], credit['class'])

# Inspect the results
sk.get_support()
# -



# +
from sklearn.ensemble.forest import RandomForestClassifier as rfc


arrh = pd.read_csv("arrh.csv")
arrh.head(5)
arrh.describe()
X = arrh.drop("class", axis=1)
y = arrh["class"]
from sklearn.model_selection import train_test_split

# Split the data into train and test, with 20% as test
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=1)

non_numeric_columns = arrh.select_dtypes(include=['object']).copy()

# Create a label encoder for each column. Encode the values
for column in non_numeric_columns:
    le = LabelEncoder()
    arrh[column] = le.fit_transform(arrh[column])

# Inspect the data types of the columns of the data frame
#print(arrh.dtypes)







# Create numeric encoding for credit_history
arrh_history_num = LabelEncoder().fit_transform(
  arrh['credit_history'])

# Create a new feature matrix including the numeric encoding
X_num = pd.concat([X, pd.Series(credit_history_num)], 1)

# Create new feature matrix with dummies for credit_history
X_hot = pd.concat(
  [X, pd.get_dummies(credit['credit_history'])], 1)

# Compare the number of features of the resulting DataFrames
X_hot.shape[1] > X_num.shape[1]

# -



# +

# Find the best value for max_depth among values 2, 5 and 10
grid_search = GridSearchCV(
  rfc(random_state=1),cv=5 , param_grid={'max_depth': [2, 5, 10]})
best_value = grid_search.fit(
  X_train, y_train).best_params_['max_depth']

# Using the best value from above, fit a random forest
clf = rfc(
  random_state=1, max_depth=best_value).fit(X_train, y_train)

# Apply SelectKBest with chi2 and pick top 100 features
vt = SelectKBest(chi2, k=100).fit(X_train, y_train)

# Create a new dataset only containing the selected features
X_train_reduced = vt.transform(X_train)

# -


