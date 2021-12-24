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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# ## The Standard Workflow

# ### Feature engineering

# +
credit = pd.read_csv('credit.csv')

# Inspect the first few lines of your data using head()
credit.head(3)
# -

non_numeric_columns = ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment', 'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker']


# +
from sklearn.preprocessing import LabelEncoder
# Create a label encoder for each column. Encode the values
for column in non_numeric_columns:
    le = LabelEncoder()
    credit[column] = le.fit_transform(credit[column])

# Inspect the data types of the columns of the data frame
print(credit.dtypes)
# -

# ### Your first pipeline

X = credit[['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings_status', 'employment', 'installment_commitment', 'personal_status', 'other_parties', 'residence_since',
       'property_magnitude', 'age', 'other_payment_plans', 'housing', 'existing_credits', 'job', 'num_dependents', 'own_telephone', 'foreign_worker']]
y = credit[['class']]

# +
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Split the data into train and test, with 20% as test
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=1)

# Create a random forest classifier, fixing the seed to 2
rf_model = RandomForestClassifier(random_state=2, n_estimators = 10).fit(
  X_train, y_train.values.ravel())

# Use it to predict the labels of the test data
rf_predictions = rf_model.predict(X_test)

# Assess the accuracy of both classifiers
accuracy_score(y_test, rf_predictions)
# -

# ### Grid search CV for model complexity

# +
from sklearn.model_selection import GridSearchCV
# Set a range for n_estimators from 10 to 40 in steps of 10
param_grid = {'n_estimators': range(10, 50, 10)}

# Optimize for a RandomForestClassifier() using GridSearchCV
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid.fit(X, y.values.ravel())
grid.best_params_

# +
from sklearn.ensemble import AdaBoostClassifier
# Define a grid for n_estimators ranging from 1 to 10
param_grid = {'n_estimators': range(1, 11)}

# Optimize for a AdaBoostClassifier() using GridSearchCV
grid = GridSearchCV(AdaBoostClassifier(), param_grid, cv=3)
grid.fit(X, y.values.ravel())
grid.best_params_

# +
from sklearn.neighbors import KNeighborsClassifier
# Define a grid for n_neighbors with values 10, 50 and 100
param_grid = {'n_neighbors': [10, 50, 100]}

# Optimize for KNeighborsClassifier() using GridSearchCV
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3)
grid.fit(X, y.values.ravel())
grid.best_params_
# -

# ### Categorical encodings

# +
# Create numeric encoding for credit_history
credit_history_num = LabelEncoder().fit_transform(
  credit['credit_history'])

# Create a new feature matrix including the numeric encoding
X_num = pd.concat([X, pd.Series(credit_history_num)], axis = 1)

# Create new feature matrix with dummies for credit_history
X_hot = pd.concat(
  [X, pd.get_dummies(credit['credit_history'])],axis = 1)

# Compare the number of features of the resulting DataFrames
X_hot.shape[1] > X_num.shape[1]
# -

# ### Feature transformations

# +
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
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

# ### Bringing it all together

# +
# Find the best value for max_depth among values 2, 5 and 10
grid_search = GridSearchCV(
   RandomForestClassifier(random_state=1, n_estimators = 10), param_grid={'max_depth': [2, 5, 10]}, cv = 3)
best_value = grid_search.fit(
  X_train, y_train.values.ravel()).best_params_['max_depth']

# Using the best value from above, fit a random forest
clf = RandomForestClassifier(
  random_state=1, max_depth=best_value, n_estimators = 10).fit(X_train, y_train.values.ravel())

# Apply SelectKBest with chi2 and pick top 100 features
vt = SelectKBest(chi2, k=5).fit(X_train, y_train.values.ravel())

# Create a new dataset only containing the selected features
X_train_reduced = vt.transform(X_train)
# -

# ## The Human in the Loop

# ### Is the source or the destination bad?

flows = pd.read_csv('lanl_flows.csv')
flows.head()


def featurize(df):
    return {
        'unique_ports': len(set(df['destination_port'])),
        'average_packet': np.mean(df['packet_count']),
        'average_duration': np.mean(df['duration'])
    }


bads = {'C332', 'C3491', 'C7503', 'C9723', 'C1', 'C1632', 'C16088', 'C8209', 'C9006', 'C1567', 'C977', 'C1448', 'C115', 'C1980', 'C3422', 'C586', 'C2085', 'C1382', 'C2725', 'C17425', 'C477', 'C113', 'C20819', 'C1484', 'C15232', 'C177', 'C504', 'C8490', 'C4610', 'C5618', 'C2816', 'C923', 'C1906', 'C1006', 'C8751', 'C1482', 'C353', 'C19444', 'C1275', 'C3455', 'C17860', 'C17600', 'C1089', 'C1966', 'C3758', 'C370', 'C1737', 'C492', 'C16401', 'C3170', 'C15', 'C1952', 'C636', 'C92', 'C1944', 'C2597', 'C1961', 'C1173', 'C108', 'C4159', 'C685', 'C1581', 'C1042', 'C20966', 'C798', 'C1611', 'C1823', 'C553', 'C305', 'C21946', 'C528', 'C3288', 'C16563', 'C2578', 'C1570', 'C583', 'C779', 'C486', 'C3629', 'C19156', 'C3521', 'C18190', 'C633', 'C5693', 'C21814', 'C1555', 'C22275', 'C2254', 'C1810', 'C1438', 'C506', 'C849', 'C10817', 'C22409', 'C10005', 'C10405', 'C346', 'C22766', 'C2877', 'C721', 'C917', 'C3888', 'C359', 'C2378', 'C2196', 'C15197', 'C882', 'C12116', 'C2944', 'C1477', 'C4403', 'C400', 'C828', 'C18025', 'C46', 'C19932', 'C1710', 'C20455', 'C1936', 'C2341', 'C1191', 'C395', 'C366', 'C3601', 'C10', 'C3037', 'C6513', 'C13713', 'C1432', 'C3755', 'C706', 'C1085', 'C11194', 'C5439', 'C2609', 'C52', 'C3173', 'C7597', 'C3305', 'C1124', 'C2846', 'C17640', 'C965', 'C801', 'C20677', 'C385', 'C5343', 'C1509', 'C513', 'C368', 'C3435', 'C102', 'C18464', 'C458', 'C625', 'C90', 'C464', 'C3635', 'C22176', 'C3388', 'C21963', 'C20203', 'C5030', 'C17806', 'C742', 'C4280', 'C398', 'C7782', 'C1065', 'C3437', 'C3813', 'C4934', 'C1125', 'C4773', 'C3597', 'C18626', 'C1022', 'C2669', 'C881', 'C2844', 'C2079', 'C687', 'C231', 'C2058', 'C11039', 'C96', 'C19038', 'C61', 'C6487', 'C1302', 'C307', 'C294', 'C1183', 'C1776', 'C11727', 'C1046', 'C1732', 'C152', 'C1461', 'C3610', 'C467', 'C791', 'C1028', 'C1119', 'C8172', 'C4161', 'C1269', 'C452', 'C17776', 'C89', 'C19803', 'C529', 'C155', 'C886', 'C1479', 'C883', 'C19356', 'C1797', 'C5653', 'C1500', 'C22174', 'C728', 'C9692', 'C306', 'C423', 'C1506', 'C1319', 'C338', 'C3380', 'C8585', 'C2013', 'C4106', 'C1626', 'C2849', 'C1015', 'C12682', 'C12512', 'C2388', 'C78', 'C313', 'C3153', 'C12448', 'C1215', 'C16467', 'C17636', 'C2914', 'C10577', 'C5111', 'C12320', 'C626', 'C2648', 'C1784', 'C1224', 'C4554', 'C1096', 'C1503', 'C21664', 'C612', 'C3586', 'C2091', 'C148', 'C2604', 'C11178', 'C3303', 'C2519', 'C2057', 'C3199', 'C126', 'C143', 'C18113', 'C3249', 'C3019', 'C429', 'C1268', 'C21919', 'C7464', 'C1415', 'C853', 'C18872', 'C765', 'C1549', 'C21349', 'C3292', 'C17693', 'C4845', 'C3699', 'C9945', 'C1610', 'C457', 'C1964', 'C1493', 'C754', 'C42', 'C1616', 'C7131', 'C5453', 'C1003', 'C430', 'C1887', 'C2012', 'C243', 'C1014', 'C302', 'C246', 'C1222'}

# +
from sklearn.model_selection import cross_val_score
# Group by source computer, and apply the feature extractor 
out = flows.groupby('source_computer').apply(featurize)

# Convert the iterator to a dataframe by calling list on it
X = pd.DataFrame(list(out), index=out.index)

# Check which sources in X.index are bad to create labels
y = [x in bads for x in X.index]

# Report the average accuracy of Adaboost over 3-fold CV
print(np.mean(cross_val_score(AdaBoostClassifier(), X, y, cv = 3)))
# -

# ### Feature engineering on grouped data

# +
# Create a feature counting unique protocols per source
protocols = flows.groupby('source_computer').apply(
  lambda df: len(set(df['protocol'])))

# Convert this feature into a dataframe, naming the column
protocols_DF = pd.DataFrame(
  protocols, index=protocols.index, columns=['protocol'])

# Now concatenate this feature with the previous dataset, X
X_more = pd.concat([X, protocols_DF], axis=1)

# Refit the classifier and report its accuracy
print(np.mean(cross_val_score(
  AdaBoostClassifier(), X_more, y, cv = 3)))
# -

# ### Turning a heuristic into a classifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# +
# Create a new dataset X_train_bad by subselecting bad hosts
X_train_bad = X_train[y_train]

# Calculate the average of unique_ports in bad examples
avg_bad_ports = np.mean(X_train_bad['unique_ports'])

# Label as positive sources that use more ports than that
pred_port = X_test['unique_ports'] > avg_bad_ports

# Print the accuracy of the heuristic
print(accuracy_score(y_test, pred_port))
# -

# ### Combining heuristics

# +
# Compute the mean of average_packet for bad sources
avg_bad_packet = np.mean(X_train[y_train]['average_packet'])

# Label as positive if average_packet is lower than that
pred_packet = X_test['average_packet'] < avg_bad_packet

# Find indices where pred_port and pred_packet both True
pred_port = X_test['unique_ports'] > avg_bad_ports
pred_both = pred_packet & pred_port

# Ports only produced an accuracy of 0.919. Is this better?
print(accuracy_score(y_test, pred_both))
# -

# ### Dealing with label noise

y_train_noisy = y_train

# +
from sklearn.naive_bayes import GaussianNB
# Fit a Gaussian Naive Bayes classifier to the training data
clf = GaussianNB().fit(X_train, y_train_noisy)

# Report its accuracy on the test data
print(accuracy_score(y_test, clf.predict(X_test)))

# Assign half the weight to the first 100 noisy examples
weights = [0.5]*100 + [1.0]*(len(y_train_noisy)-100)

# Refit using weights and report accuracy. Has it improved?
clf_weights = GaussianNB().fit(X_train, y_train_noisy, sample_weight=weights)
print(accuracy_score(y_test, clf_weights.predict(X_test)))
# -

# ### Reminder of performance metrics

from sklearn.metrics import f1_score, precision_score, confusion_matrix
print(f1_score(y_test, clf.predict(X_test)))
print(precision_score(y_test, clf.predict(X_test)))

# ### Real-world cost analysis

# +
# Fit a random forest classifier to the training data
clf = RandomForestClassifier(random_state=2, n_estimators = 10).fit(X_train, y_train)

# Label the test data
preds = clf.predict(X_test)

# Get false positives/negatives from the confusion matrix
tp, fp, fn, tn = confusion_matrix(y_test, preds).flatten()

# Now compute the cost using the manager's advice
cost = fp*10000 + fn*150000
# -

# ### Default thresholding

# +
# Score the test data using the given classifier
scores = clf.predict_proba(X_test)

# Get labels from the scores using the default threshold
preds = [s[1] > 0.5 for s in scores]

# Use the predict method to label the test data again
preds_default = clf.predict(X_test)

# Compare the two sets of predictions
all(preds == preds_default)
# -

# ### Optimizing the threshold

# +
# Create a range of equally spaced threshold values
t_range = [0.0, 0.25, 0.5, 0.75, 1.0]

# Store the predicted labels for each value of the threshold
preds = [[s[1] > thr for s in scores] for thr in t_range]

# Compute the accuracy for each threshold
accuracies = [accuracy_score(y_test, p) for p in preds]

# Compute the F1 score for each threshold
f1_scores = [f1_score(y_test, p) for p in preds]

# Report the optimal threshold for accuracy, and for F1
print(t_range[np.argmax(accuracies)], t_range[np.argmax(f1_scores)])
# -

# ### Bringing it all together

arrh = pd.read_csv('arrh.csv')
X = arrh[['age', 'sex', 'height', 'weight', 'QRSduration', 'PRinterval', 'Q-Tinterval', 'Tinterval', 'Pinterval', 'QRS', 'T', 'P', 'QRST', 'J', 'heartrate', 'chDI_Qwave', 'chDI_Rwave', 'chDI_Swave', 'chDI_RPwave', 'chDI_SPwave', 'chDI_intrinsicReflecttions', 'chDI_RRwaveExists', 'chDI_DD_RRwaveExists', 'chDI_RPwaveExists', 'chDI_DD_RPwaveExists', 'chDI_RTwaveExists', 'chDI_DD_RTwaveExists', 'chDII_Qwave', 'chDII_Rwave', 'chDII_Swave', 'chDII_RPwave', 'chDII_SPwave', 'chDII_intrinsicReflecttions', 'chDII_RRwaveExists', 'chDII_DD_RRwaveExists', 'chDII_RPwaveExists', 'chDII_DD_RPwaveExists', 'chDII_RTwaveExists', 'chDII_DD_RTwaveExists', 'chDIII_Qwave', 'chDIII_Rwave', 'chDIII_Swave', 'chDIII_RPwave', 'chDIII_SPwave', 'chDIII_intrinsicReflecttions', 'chDIII_RRwaveExists', 'chDIII_DD_RRwaveExists', 'chDIII_RPwaveExists', 'chDIII_DD_RPwaveExists', 'chDIII_RTwaveExists', 'chDIII_DD_RTwaveExists', 'chAVR_Qwave', 'chAVR_Rwave', 'chAVR_Swave', 'chAVR_RPwave', 'chAVR_SPwave', 'chAVR_intrinsicReflecttions', 'chAVR_RRwaveExists', 'chAVR_DD_RRwaveExists', 'chAVR_RPwaveExists', 'chAVR_DD_RPwaveExists', 'chAVR_RTwaveExists', 'chAVR_DD_RTwaveExists', 'chAVL_Qwave', 'chAVL_Rwave', 'chAVL_Swave', 'chAVL_RPwave', 'chAVL_SPwave', 'chAVL_intrinsicReflecttions', 'chAVL_RRwaveExists', 'chAVL_DD_RRwaveExists', 'chAVL_RPwaveExists', 'chAVL_DD_RPwaveExists', 'chAVL_RTwaveExists', 'chAVL_DD_RTwaveExists', 'chAVF_Qwave', 'chAVF_Rwave', 'chAVF_Swave', 'chAVF_RPwave', 'chAVF_SPwave', 'chAVF_intrinsicReflecttions', 'chAVF_RRwaveExists', 'chAVF_DD_RRwaveExists', 'chAVF_RPwaveExists', 'chAVF_DD_RPwaveExists', 'chAVF_RTwaveExists', 'chAVF_DD_RTwaveExists', 'chV1_Qwave', 'chV1_Rwave', 'chV1_Swave', 'chV1_RPwave', 'chV1_SPwave', 'chV1_intrinsicReflecttions', 'chV1_RRwaveExists', 'chV1_DD_RRwaveExists', 'chV1_RPwaveExists', 'chV1_DD_RPwaveExists', 'chV1_RTwaveExists', 'chV1_DD_RTwaveExists', 'chV2_Qwave', 'chV2_Rwave', 'chV2_Swave', 'chV2_RPwave', 'chV2_SPwave', 'chV2_intrinsicReflecttions', 'chV2_RRwaveExists', 'chV2_DD_RRwaveExists', 'chV2_RPwaveExists', 'chV2_DD_RPwaveExists', 'chV2_RTwaveExists', 'chV2_DD_RTwaveExists', 'chV3_Qwave', 'chV3_Rwave', 'chV3_Swave', 'chV3_RPwave', 'chV3_SPwave', 'chV3_intrinsicReflecttions', 'chV3_RRwaveExists', 'chV3_DD_RRwaveExists', 'chV3_RPwaveExists', 'chV3_DD_RPwaveExists', 'chV3_RTwaveExists', 'chV3_DD_RTwaveExists', 'chV4_Qwave', 'chV4_Rwave', 'chV4_Swave', 'chV4_RPwave', 'chV4_SPwave', 'chV4_intrinsicReflecttions', 'chV4_RRwaveExists', 'chV4_DD_RRwaveExists', 'chV4_RPwaveExists', 'chV4_DD_RPwaveExists', 'chV4_RTwaveExists', 'chV4_DD_RTwaveExists', 'chV5_Qwave', 'chV5_Rwave', 'chV5_Swave', 'chV5_RPwave', 'chV5_SPwave', 'chV5_intrinsicReflecttions', 'chV5_RRwaveExists', 'chV5_DD_RRwaveExists', 'chV5_RPwaveExists', 'chV5_DD_RPwaveExists', 'chV5_RTwaveExists', 'chV5_DD_RTwaveExists', 'chV6_Qwave', 'chV6_Rwave', 'chV6_Swave', 'chV6_RPwave', 'chV6_SPwave', 'chV6_intrinsicReflecttions', 'chV6_RRwaveExists', 'chV6_DD_RRwaveExists', 'chV6_RPwaveExists', 'chV6_DD_RPwaveExists', 'chV6_RTwaveExists', 'chV6_DD_RTwaveExists', 'chDI_JJwaveAmp', 'chDI_QwaveAmp', 'chDI_RwaveAmp', 'chDI_SwaveAmp', 'chDI_RPwaveAmp', 'chDI_SPwaveAmp', 'chDI_PwaveAmp', 'chDI_TwaveAmp', 'chDI_QRSA', 'chDI_QRSTA', 'chDII_JJwaveAmp', 'chDII_QwaveAmp', 'chDII_RwaveAmp', 'chDII_SwaveAmp', 'chDII_RPwaveAmp', 'chDII_SPwaveAmp', 'chDII_PwaveAmp', 'chDII_TwaveAmp', 'chDII_QRSA', 'chDII_QRSTA', 'chDIII_JJwaveAmp', 'chDIII_QwaveAmp', 'chDIII_RwaveAmp', 'chDIII_SwaveAmp', 'chDIII_RPwaveAmp', 'chDIII_SPwaveAmp', 'chDIII_PwaveAmp', 'chDIII_TwaveAmp', 'chDIII_QRSA', 'chDIII_QRSTA', 'chAVR_JJwaveAmp', 'chAVR_QwaveAmp', 'chAVR_RwaveAmp', 'chAVR_SwaveAmp', 'chAVR_RPwaveAmp', 'chAVR_SPwaveAmp', 'chAVR_PwaveAmp', 'chAVR_TwaveAmp', 'chAVR_QRSA', 'chAVR_QRSTA', 'chAVL_JJwaveAmp', 'chAVL_QwaveAmp', 'chAVL_RwaveAmp', 'chAVL_SwaveAmp', 'chAVL_RPwaveAmp', 'chAVL_SPwaveAmp', 'chAVL_PwaveAmp', 'chAVL_TwaveAmp', 'chAVL_QRSA', 'chAVL_QRSTA', 'chAVF_JJwaveAmp', 'chAVF_QwaveAmp', 'chAVF_RwaveAmp', 'chAVF_SwaveAmp', 'chAVF_RPwaveAmp', 'chAVF_SPwaveAmp', 'chAVF_PwaveAmp', 'chAVF_TwaveAmp', 'chAVF_QRSA', 'chAVF_QRSTA', 'chV1_JJwaveAmp', 'chV1_QwaveAmp', 'chV1_RwaveAmp', 'chV1_SwaveAmp', 'chV1_RPwaveAmp', 'chV1_SPwaveAmp', 'chV1_PwaveAmp', 'chV1_TwaveAmp', 'chV1_QRSA', 'chV1_QRSTA', 'chV2_JJwaveAmp', 'chV2_QwaveAmp', 'chV2_RwaveAmp', 'chV2_SwaveAmp', 'chV2_RPwaveAmp', 'chV2_SPwaveAmp', 'chV2_PwaveAmp', 'chV2_TwaveAmp', 'chV2_QRSA', 'chV2_QRSTA', 'chV3_JJwaveAmp', 'chV3_QwaveAmp', 'chV3_RwaveAmp', 'chV3_SwaveAmp', 'chV3_RPwaveAmp', 'chV3_SPwaveAmp', 'chV3_PwaveAmp', 'chV3_TwaveAmp', 'chV3_QRSA', 'chV3_QRSTA', 'chV4_JJwaveAmp', 'chV4_QwaveAmp', 'chV4_RwaveAmp', 'chV4_SwaveAmp', 'chV4_RPwaveAmp', 'chV4_SPwaveAmp', 'chV4_PwaveAmp', 'chV4_TwaveAmp', 'chV4_QRSA', 'chV4_QRSTA', 'chV5_JJwaveAmp', 'chV5_QwaveAmp', 'chV5_RwaveAmp', 'chV5_SwaveAmp', 'chV5_RPwaveAmp', 'chV5_SPwaveAmp', 'chV5_PwaveAmp', 'chV5_TwaveAmp', 'chV5_QRSA', 'chV5_QRSTA', 'chV6_JJwaveAmp', 'chV6_QwaveAmp', 'chV6_RwaveAmp', 'chV6_SwaveAmp', 'chV6_RPwaveAmp', 'chV6_SPwaveAmp', 'chV6_PwaveAmp', 'chV6_TwaveAmp', 'chV6_QRSA', 'chV6_QRSTA']]
y = arrh[['class']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
arrh.head()

# +
from sklearn.tree import DecisionTreeClassifier
# Create a scorer assigning more cost to false positives
def my_scorer(y_test, y_est, cost_fp=10.0, cost_fn=1.0):
    tn, fp, fn, tp = confusion_matrix(y_test, y_est).ravel()
    return cost_fp*fp + cost_fn*fn

# Fit a DecisionTreeClassifier to the data and compute the loss
clf = DecisionTreeClassifier(random_state=2).fit(X_train, y_train)
print(my_scorer(y_test, clf.predict(X_test)))

# Refit, downweighting subjects whose weight is above 80
weights = [0.5 if w > 80 else 1.0 for w in X_train.weight]
clf_weighted = DecisionTreeClassifier().fit(
  X_train, y_train, sample_weight=weights)
print(my_scorer(y_test, clf_weighted.predict(X_test)))
# -

# ## Model Lifecycle Management
#

# ### Your first pipeline - again!

# +
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif
import warnings 
warnings.filterwarnings('ignore')

# Create pipeline with feature selector and classifier
pipe = Pipeline([
    ('feature_selection', SelectKBest(f_classif)),
    ('clf', RandomForestClassifier(random_state=2))])

# Create a parameter grid
params = {
   'feature_selection__k':[10, 20],
   'clf__n_estimators':[2, 5]}

# Initialise the grid search object
grid_search = GridSearchCV(pipe, param_grid=params, cv = 3)

# Fit it to the data and print the best value combination
print(grid_search.fit(X_train, y_train).best_params_)
# -

# ### Custom scorers in pipelines

# +
from sklearn.metrics import roc_auc_score, make_scorer

# Create a custom scorer
scorer = make_scorer(roc_auc_score)

# Initialize the CV object
gs = GridSearchCV(pipe, param_grid=params, scoring=scorer, cv = 3)

# Fit it to the data and print the winning combination
print(gs.fit(X_train, y_train).best_params_)

# +
# Create a custom scorer
scorer = make_scorer(f1_score)

# Initialise the CV object
gs = GridSearchCV(pipe, param_grid=params, scoring=scorer)

# Fit it to the data and print the winning combination
print(gs.fit(X_train, y_train).best_params_)


# -

def my_metric(y_test, y_est, cost_fp=10.0, cost_fn=1.0):
    tn, fp, fn, tp = confusion_matrix(y_test, y_est).ravel()
    return cost_fp * fp + cost_fn * fn


# +
# Create a custom scorer
scorer = make_scorer(my_metric)

# Initialise the CV object
gs = GridSearchCV(pipe, param_grid=params, scoring=scorer)

# Fit it to the data and print the winning combination
print(gs.fit(X_train, y_train).best_params_)
# -

# ### Pickles

# +
import pickle
# Fit a random forest to the training set
clf = RandomForestClassifier(random_state=42).fit(
  X_train, y_train)

# Save it to a file, to be pushed to production
with open('model.pkl', 'wb') as file:
    pickle.dump(clf, file=file)

# Now load the model from file in the production environment
with open('model.pkl', 'rb') as file:
    clf_from_file = pickle.load(file)

# Predict the labels of the test dataset
preds = clf_from_file.predict(X_test)
# -

# ### Custom function transformers in pipelines

# +
from sklearn.preprocessing import FunctionTransformer
# Define a feature extractor to flag very large values
def more_than_average(X, multiplier=1.0):
    Z = X.copy()
    Z[:,1] = Z[:,1] > multiplier*np.mean(Z[:,1])
    return Z

# Convert your function so that it can be used in a pipeline
pipe = Pipeline([
  ('ft', FunctionTransformer(more_than_average)),
  ('clf', RandomForestClassifier(random_state=2))])

# Optimize the parameter multiplier using GridSearchCV
params = {'ft__multiplier':[1, 2, 3]}
grid_search = GridSearchCV(pipe, param_grid=params)
# -

# ### Challenge the champion

# +
# Load the current model from disk
champion = pickle.load(open('model.pkl', 'rb'))

# Fit a Gaussian Naive Bayes to the training data
challenger = GaussianNB().fit(X_train, y_train)

# Print the F1 test scores of both champion and challenger
print(f1_score(y_test, champion.predict(X_test)))
print(f1_score(y_test, challenger.predict(X_test)))

# Write back to disk the best-performing model
with open('model.pkl', 'wb') as file:
    pickle.dump(champion, file=file)
# -

# ### Cross-validation statistics

# +
pipe = Pipeline([
    ('feature_selection', SelectKBest(f_classif)),
    ('clf', RandomForestClassifier(random_state=2))])

params = {'feature_selection__k': [10, 20], 'clf__n_estimators': [2, 5]}

pipe = Pipeline([
    ('feature_selection', SelectKBest(k=10)), 
     ('clf', RandomForestClassifier())])

# Fit your pipeline using GridSearchCV with three folds
grid_search = GridSearchCV(pipe, params, cv=3, return_train_score=True)

# Fit the grid search
gs = grid_search.fit(X_train, y_train)

# Store the results of CV into a pandas dataframe
results = pd.DataFrame(gs.cv_results_)

# Print the difference between mean test and training scores
print(results['mean_test_score']-results['mean_train_score'])
# -

# ### Tuning the window size

# +
wrange = range(10, 100, 10)
t_now = 400
               
# Loop over window sizes
for w_size in wrange:

    # Define sliding window
    sliding = arrh.loc[(t_now - w_size + 1):t_now]

    # Extract X and y from the sliding window
    X, y = sliding.drop('class', 1), sliding['class']
    
    # Fit the classifier and store the F1 score
    preds = GaussianNB().fit(X, y).predict(X_test)
    accuracies.append(f1_score(y_test, preds))

# Estimate the best performing window size
optimal_window = wrange[np.argmax(accuracies)]
# -

# ### Bringing it all together

# +
# Create a pipeline 
pipe = Pipeline([
  ('ft', SelectKBest()), ('clf', RandomForestClassifier(random_state=2))])

# Create a parameter grid
grid = {'ft__k':[5, 10], 'clf__max_depth':[10, 20]}

# Execute grid search CV on a dataset containing under 50s
grid_search = GridSearchCV(pipe, param_grid=grid)
arrh = arrh.iloc[np.where(arrh['age'] < 50)]
grid_search.fit(arrh.drop('class', 1), arrh['class'])

# Push the fitted pipeline to production
with open('pipe.pkl', 'wb') as file:
    pickle.dump(grid_search, file)
# -

# ## Unsupervised Workflows

# ### A simple outlier

# +
# Import the LocalOutlierFactor module
from sklearn.neighbors import LocalOutlierFactor as lof

# Create the list [1.0, 1.0, ..., 1.0, 10.0] as explained
x = [1.0]*30
x.append(10)

# Cast to a data frame
X = pd.DataFrame(x)

# Fit the local outlier factor and print the outlier scores
print(lof().fit_predict(X))
# -

# ### LoF contamination

# +
# Fit the local outlier factor and output predictions
preds = lof().fit_predict(X_test)

# Print the confusion matrix
print(confusion_matrix(y_test, preds))

# +
# Set the contamination parameter to 0.2
preds = lof(contamination=0.2).fit_predict(X_test)

# Print the confusion matrix
print(confusion_matrix(y_test, preds))

# +
# # Contamination to match outlier frequency in ground_truth
# preds = lof(contamination=np.mean(y_test==-1.0)).fit_predict(X_test)

# # Print the confusion matrix
# print(confusion_matrix(y_test, preds))
# -

# ### A simple novelty

# +
# Create a list of thirty 1s and cast to a dataframe
X = pd.DataFrame([1.0]*30)

# Create an instance of a lof novelty detector
detector = lof(novelty=True)

# Fit the detector to the data
detector.fit(X)

# Use it to predict the label of an example with value 10.0
print(detector.predict(pd.DataFrame([10.0])))
# -

# ### Three novelty detectors

# +
# Import the novelty detector
from sklearn.svm import OneClassSVM as onesvm

# Fit it to the training data and score the test data
svm_detector = onesvm().fit(X_train)
scores = svm_detector.score_samples(X_test)

# +
# Import the novelty detector

from sklearn.ensemble import IsolationForest as isof

# Fit it to the training data and score the test data
isof_detector = isof().fit(X_train)
scores = isof_detector.score_samples(X_test)

# +
# Import the novelty detector
from sklearn.neighbors import LocalOutlierFactor as lof

# Fit it to the training data and score the test data
lof_detector = lof(novelty=True).fit(X_train)
scores = lof_detector.score_samples(X_test)
# -

# ### Contamination revisited

# +
# Fit a one-class SVM detector and score the test data
nov_det = onesvm().fit(X_train)
scores = nov_det.score_samples(X_test)

# Find the observed proportion of outliers in the test data
prop = np.mean(y_test==1.0)

# Compute the appropriate threshold
threshold = np.quantile(scores, prop)

# Print the confusion matrix for the thresholded scores
print(confusion_matrix(y_test, scores > threshold))
# -

# ### Find the neighbor

hep = pd.read_csv('hep.csv')
hep.head()

features = hep[['AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA', 'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT',
       'ALBUMIN', 'PROTIME', 'HISTOLOGY']]

# +
# Import DistanceMetric as dm
from sklearn.neighbors import DistanceMetric as dm

# Find the Euclidean distance between all pairs
dist_eucl = dm.get_metric('euclidean').pairwise(features)

# Find the Hamming distance between all pairs
dist_hamm = dm.get_metric('hamming').pairwise(features)

# Find the Chebyshev distance between all pairs
dist_cheb = dm.get_metric('chebyshev').pairwise(features)
# -

# ### Not all metrics agree

# +
# Compute outliers according to the euclidean metric
out_eucl = lof(metric='euclidean').fit_predict(features)

# Compute outliers according to the hamming metric
out_hamm = lof(metric='hamming').fit_predict(features)

# Compute outliers according to the jaccard metric
out_jacc  = lof(metric='jaccard').fit_predict(features)

# Find if the metrics agree on any one datapoint
print(any(out_eucl+out_hamm+out_jacc ==-3 ))
# -

# ### Restricted Levenshtein

proteins = pd.read_csv('proteins_exercises.csv')
proteins.head()

# +
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
import stringdist
# Wrap the RD-Levenshtein metric in a custom function
def my_rdlevenshtein(u, v):
    return stringdist.rdlevenshtein(u[0], v[0])

# Reshape the array into a numpy matrix
sequences = np.array(proteins['seq']).reshape(-1, 1)

# Compute the pairwise distance matrix in square form
M = squareform(pdist(sequences, my_rdlevenshtein))

# Run a LoF algorithm on the precomputed distance matrix
preds = lof(metric='precomputed').fit_predict(M)

# Compute the accuracy of the outlier predictions
print(accuracy_score(proteins['label'] == 'VIRUS', preds == -1))
# -

# ### Bringing it all together

# +
from sklearn.metrics import roc_auc_score
# Create a feature that contains the length of the string
proteins['len'] = proteins['seq'].apply(lambda x: len(x))

# Create a feature encoding the first letter of the string
proteins['first'] =  LabelEncoder().fit_transform(
  proteins['seq'].apply(lambda x: list(x)[0]))

lof_detector = lof(novelty=True).fit(proteins[['len', 'first']])

# Extract scores from the fitted LoF object, compute its AUC
scores_lof = lof_detector.negative_outlier_factor_
print(roc_auc_score(proteins['label']=='IMMUNE SYSTEM', scores_lof))

# Fit a 1-class SVM, extract its scores, and compute its AUC
svm = onesvm().fit(proteins[['len', 'first']])
scores_svm = svm.score_samples(proteins[['len', 'first']])
print(roc_auc_score(proteins['label']=='IMMUNE SYSTEM', scores_svm))
