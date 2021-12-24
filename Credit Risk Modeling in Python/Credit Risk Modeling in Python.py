# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [raw]
# 1 - Exploring and Preparing Loan Data
#
# In this first chapter, we will discuss the concept of credit risk and define how it is calculated. Using cross tables and plots, we will explore a real-world data set. Before applying machine learning, we will process this data by finding and resolving problems.

# %%

# %%
## Explore the credit data

# Look at the distribution of loan amounts with a histogram
n, bins, patches = plt.hist(x=cr_loan['loan_amnt'], bins='auto', color='blue',alpha=0.7, rwidth=0.85)
plt.xlabel("Loan Amount")
plt.show()

_________________________________________________________

print("There are 32 000 rows of data so the scatter plot may take a little while to plot.")

# Plot a scatter plot of income against age
plt.scatter(cr_loan['person_income'], cr_loan['person_age'],c='blue', alpha=0.5)

plt.xlabel('Personal Income')
plt.ylabel('Persone Age')
plt.show()

# %%

# %%
## Crosstab and pivot tables

# Create a cross table of the loan intent and loan status
print(pd.crosstab(cr_loan['loan_intent'], cr_loan['loan_status'], margins = True))
___________________________________________________________________

# Create a cross table of home ownership, loan status, and grade
print(pd.crosstab(cr_loan['person_home_ownership'], [cr_loan['loan_status'], cr_loan['loan_grade']]))
___________________________________________________________________

# Create a cross table of home ownership, loan status, and average percent income
print(pd.crosstab(cr_loan['person_home_ownership'], cr_loan['loan_status'],
              values=cr_loan['loan_percent_income'], aggfunc='mean'))
___________________________________________________________________

# Create a box plot of percentage income by loan status
cr_loan.boxplot(column = ['loan_percent_income'], by = 'loan_status')
plt.title('Average Percent Income by Loan Status')
plt.suptitle('')
plt.show()

# %%

# %%
## Finding outliers with cross tables

# Create the cross table for loan status, home ownership, and the max employment length
print(pd.crosstab(cr_loan['loan_status'],cr_loan['person_home_ownership'],
                  values=cr_loan['person_emp_length'], aggfunc='max'))
___________________________________________________________________

# Create an array of indices where employment length is greater than 60
indices = cr_loan[cr_loan['person_emp_length'] > 60].index

____________________________________________________________________

# Drop the records from the data based on the indices and create a new dataframe
cr_loan_new = cr_loan.drop(indices)
____________________________________________________________________

# Create the cross table from earlier and include minimum employment length
print(pd.crosstab(cr_loan_new['loan_status'], cr_loan_new['person_home_ownership'],
            values=cr_loan_new['person_emp_length'], aggfunc=['min','max']))



# %%

# %%
## Visualizing credit outliers

# Create the scatter plot for age and amount
plt.scatter(cr_loan['person_age'],
            cr_loan['loan_amnt'], c='blue', alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Amount")
plt.show()

_________________________________________________

# Use Pandas to drop the record from the data frame and create a new one
cr_loan_new = cr_loan.drop(cr_loan[cr_loan['person_age'] > 100].index)

# Create a scatter plot of age and interest rate
colors = ["blue","red"]
plt.scatter(cr_loan_new['person_age'], cr_loan_new['loan_int_rate'],
            c = cr_loan_new['loan_status'],
            cmap = matplotlib.colors.ListedColormap(colors),
            alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Interest Rate")
plt.show()

# %%

# %%
## Replacing missing credit data

# Print a null value column array
print(cr_loan.columns[cr_loan.isnull().any()])

# Print the top five rows with nulls for employment length
print(cr_loan[cr_loan['person_emp_length'].isnull()].head())

# Impute the null values with the median value for all employment lengths
cr_loan['person_emp_length'].fillna((cr_loan['person_emp_length'].median()), inplace=True)

# Create a histogram of employment length
n, bins, patches = plt.hist(cr_loan['person_emp_length'], bins='auto', color='blue')
plt.xlabel("Person Employment Length")
plt.show()

# %%

# %%
## Removing missing data

# Print the number of nulls
print(cr_loan['loan_int_rate'].isnull().sum())

# Store the array on indices
indices = cr_loan[cr_loan['loan_int_rate'].isnull()].index

# Save the new data without missing data
cr_loan_clean = cr_loan.drop(indices)

# %%

# %%
## Missing data intuition

cr_loan['person_home_ownership'].value_counts(dropna=False)

# Replace the data with the value Other.

# %%

# %% [raw]
# 2 - Logistic Regression for Defaults
#
# With the loan data fully prepared, we will discuss the logistic regression model which is a standard in risk modeling. We will understand the components of this model as well as how to score its performance. Once we've created predictions, we can explore the financial impact of utilizing this model.

# %%

# %%
## Logistic regression basics

# Create the X and y data sets
X = cr_loan_clean[['loan_int_rate']]
y = cr_loan_clean[['loan_status']]

# Create and fit a logistic regression model
clf_logistic_single = LogisticRegression()
clf_logistic_single.fit(X, np.ravel(y))

# Print the parameters of the model
print(clf_logistic_single.get_params())

# Print the intercept of the model
print(clf_logistic_single.intercept_)

# %%

# %%
## Multivariate logistic regression

# Create X data for the model
X_multi = cr_loan_clean[['loan_int_rate', 'person_emp_length']]

# Create a set of y data for training
y = cr_loan_clean[['loan_status']]

# Create and train a new logistic regression
clf_logistic_multi = LogisticRegression(solver='lbfgs').fit(X_multi, np.ravel(y))

# Print the intercept of the model
print(clf_logistic_multi.intercept_)


# <script.py> output:
#     [-4.21645549]

# O novo modelo clf_logistic_multi possui um valor .intercept_ 
# mais próximo de zero. Isso significa que as probabilidades de 
# log de um não padrão estão se aproximando de zero
__________________________________________

# This means the log odds of a non-default is approaching zero.

# %%

# %%
## Creating training and test sets

# Create the X and y data sets
X = cr_loan_clean[['loan_int_rate','person_emp_length', 'person_income']]
y = cr_loan_clean[['loan_status']]

# Use test_train_split to create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)

# Create and fit the logistic regression model
clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

# Print the models coefficients
print(clf_logistic.coef_)

# <script.py> output:
#     [[ 1.28517496e-09 -2.27622202e-09 -2.17211991e-05]]

# Do you see that three columns were used for training and there 
# are three values in .coef_? This tells you how important each 
# column, or feature, was for predicting. The more positive 
# the value, the more it predicts defaults. 
# Look at the value for loan_int_rate.

# %%

# %%
## Changing coefficients

# Print the first five rows of each training set
print(X1_train.head())
print(X2_train.head())

# Create and train a model on the first training data
clf_logistic1 = LogisticRegression(solver='lbfgs').fit(X1_train, np.ravel(y_train))

# Create and train a model on the second training data
clf_logistic2 = LogisticRegression(solver='lbfgs').fit(X2_train, np.ravel(y_train))

# Print the coefficients of each model
print(clf_logistic1.coef_)
print(clf_logistic2.coef_)

# %%

# %%
## One-hot encoding credit data

# Create two data sets for numeric and non-numeric data
cred_num = cr_loan_clean.select_dtypes(exclude=['object'])
cred_str = cr_loan_clean.select_dtypes(include=['object'])

# One-hot encode the non-numeric columns
cred_str_onehot = pd.get_dummies(cred_str)

# Union the one-hot encoded columns to the numeric ones
cr_loan_prep = pd.concat([cred_num, cred_str_onehot], axis=1)

# Print the columns in the new data set
print(cr_loan_prep.columns)

# If you've ever seen a credit scorecard, the column_name_value 
# format should look familiar. If you haven't seen one, look up
# some pictures during your next break!

# %%

# %%
## Predicting probability of default

# Train the logistic regression model on the training data
clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

# Create predictions of probability for loan status using test data
preds = clf_logistic.predict_proba(X_test)

# Create dataframes of first five predictions, and first five true labels
preds_df = pd.DataFrame(preds[:,1][0:5], columns = ['prob_default'])
true_df = y_test.head()

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1))

# %%

# %%
## Default classification reporting

# Create a dataframe for the probabilities of default
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_default'])

# Reassign loan status based on the threshold
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > .5 else 0)

# Print the row counts for each loan status
print(preds_df['loan_status'].value_counts())

# Print the classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))

# %%

# %%
## Selecting report metrics

# Print the classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))
_________________________________________________

# Print all the non-average values from the report
print(precision_recall_fscore_support(y_test, preds_df['loan_status']))

_________________________________________________

# Print the first two numbers from the report
print(precision_recall_fscore_support(y_test, preds_df['loan_status'])[:2])


# %%

# %%
## Visually scoring credit models

# Create predictions and store them in a variable
preds = clf_logistic.predict_proba(X_test)

# Print the accuracy score the model
print(clf_logistic.score(X_test, y_test))

# Plot the ROC curve of the probabilities of default
prob_default = preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(y_test, prob_default)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.show()

# Compute the AUC and store it in a variable
auc = roc_auc_score(y_test, prob_default)

## ROC AUC PLOT

# what the ROC chart shows us is the tradeoff between
# all values of our false positive rate (fallout) and true positive rate (sensitivity).

# %%

# %%
## Thresholds and confusion matrices

# Set the threshold for defaults to 0.5
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.5 else 0)

# Print the confusion matrix
print(confusion_matrix(y_test, preds_df['loan_status']))

# <script.py> output:
#     [[9023  175]
#      [2152  434]]
_______________________________________________________

# Set the threshold for defaults to 0.4
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > .4 else 0)

# Print the confusion matrix
print(confusion_matrix(y_test, preds_df['loan_status']))

# <script.py> output:
#     [[8476  722]
#      [1386 1200]]

# %%

# %%
## How thresholds affect performance

# Reassign the values of loan status based on the new threshold
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > .4 else 0)

# Store the number of loan defaults from the prediction data
num_defaults = preds_df['loan_status'].value_counts()[1]

# Store the default recall from the classification report
default_recall = precision_recall_fscore_support(y_test, preds_df['loan_status'])[1][1]

# Calculate the estimated impact of the new default recall rate
print(avg_loan_amnt * num_defaults * (1 - default_recall))

____________________________________________________________

# Loan Amount     Defaults Predicted / Not Predicted        Estimated Loss on Defaults
#     $50                      .04 / .96                    (50000 x .96) x 50 = $2,400,000
____________________________________________________________

# By our estimates, this loss would be around $9.8 million. 
# That seems like a lot! Try rerunning this code with threshold 
# values of 0.3 and 0.5. Do you see the estimated losses changing? 
# How do we find a good threshold value based on these metrics alone?

# %%

# %%
## Threshold selection

plt.plot(thresh, def_recalls)
plt.plot(thresh, nondef_recalls)
plt.plot(thresh, accs)
plt.xlabel("Probability Threshold")
plt.xticks(ticks)
plt.legend(["Default Recall","Non-default Recall","Model Accuracy"])
plt.show()


# %%

# %% [raw]
# 3 - Gradient Boosted Trees Using XGBoos

# %%

# %%
# Train a model
import xgboost as xgb
clf_gbt = xgb.XGBClassifier().fit(X_train, np.ravel(y_train))

# Predict with a model
gbt_preds = clf_gbt.predict_proba(X_test)

# Create dataframes of first five predictions, and first five true labels
preds_df = pd.DataFrame(gbt_preds[:,1][0:5], columns = ['prob_default'])
true_df = y_test.head()

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1))

# %%

# %%
## Gradient boosted portfolio performance

# %% [raw]
#    gbt_prob_default  lr_prob_default  lgd  loan_amnt
# 0          0.940435         0.445779  0.2      15000
# 1          0.922014         0.223447  0.2      11200
# 2          0.021707         0.288558  0.2      15000
# 3          0.026483         0.169358  0.2      10800
# 4          0.064803         0.114182  0.2       3000

# %%

# Print the first five rows of the portfolio data frame
print(portfolio.head())

# Create expected loss columns for each model using the formula
portfolio['gbt_expected_loss'] = portfolio['gbt_prob_default'] * portfolio['lgd'] * portfolio['loan_amnt']
portfolio['lr_expected_loss'] = portfolio['lr_prob_default'] * portfolio['lgd'] * portfolio['loan_amnt']

# Print the sum of the expected loss for lr
print('LR expected loss: ', np.sum(portfolio['lr_expected_loss']))

# Print the sum of the expected loss for gbt
print('GBT expected loss: ', np.sum(portfolio['gbt_expected_loss']))


#    LR expected loss:  5596776.980
#    GBT expected loss:  5447712.942

# %% [raw]
# A data frame called portfolio has been created to combine the probabilities of default for both models, the loss given default (assume 20% for now), and the loan_amnt which will be assumed to be the exposure at default
#
# Formula
# Expected Loss = PD(prob_default) * LGD(loss given default) * Loan_amount (or mean amount) 

# %% [raw]
# LGB = Loss given default or LGD is the share of an asset that is lost if a borrower defaults. It is a common parameter in risk models and also a parameter used in the calculation of economic capital, expected loss or regulatory capital under Basel II for a banking institution.

# %%

# %%
## Assessing gradient boosted trees

# Predict the labels for loan status
gbt_preds = clf_gbt.predict(X_test)

# Check the values created by the predict method
print(gbt_preds)

# Print the classification report of the model
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, gbt_preds, target_names=target_names))

# %%

# %%
## Column importance and default prediction

# Create and train the model on the training data
clf_gbt = xgb.XGBClassifier().fit(X_train, np.ravel(y_train))

# Print the column importances from the model
print(clf_gbt.get_booster().get_score(importance_type = 'weight'))


# <script.py> output:
#     {'loan_percent_income': 121, 'loan_int_rate': 183, 'person_income': 278, 
#      'person_home_ownership_MORTGAGE': 39, 'loan_amnt': 47, 'loan_grade_F': 6}
_____________________________________________________________

# So, the importance for loan_grade_F is only 6 in this case. 
# This could be because there are so few of the F-grade loans. 
# While the F-grade loans don't add much to predictions here, 
# they might affect the importance of other training columns.

# %%

# %%
## Visualizing column importance

# Train a model on the X data with 2 columns
clf_gbt2 = xgb.XGBClassifier().fit(X2_train, np.ravel(y_train))

# Plot the column importance for this model
xgb.plot_importance(clf_gbt2, importance_type = 'weight')
plt.show()
________________________________________________________

# Train a model on the X data with 3 columns
clf_gbt3 = xgb.XGBClassifier().fit(X3_train,np.ravel(y_train))

# Plot the column importance for this model
xgb.plot_importance(clf_gbt3, importance_type = 'weight')
plt.show()



# %%

# %%
## Column selection and model performance

# Predict the loan_status using each model
gbt_preds = gbt.predict(X_test)
gbt2_preds = gbt2.predict(X2_test)

# Print the classification report of the first model
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, gbt_preds, target_names=target_names))

# Print the classification report of the second model
print(classification_report(y_test, gbt2_preds, target_names=target_names))

# %%

# %%
## Cross validating credit models

# Set the values for number of folds and stopping iterations
n_folds = 5
early_stopping = 10

# Create the DTrain matrix for XGBoost
DTrain = xgb.DMatrix(X_train, label = y_train)

# Create the data frame of cross validations
cv_df = xgb.cv(params, DTrain, num_boost_round = 5, nfold=n_folds,
            early_stopping_rounds=early_stopping)

# Print the cross validations data frame
print(cv_df)

# %% [raw]
# <script.py> output:
#        train-auc-mean  train-auc-std  test-auc-mean  test-auc-std
#     0        0.898182       0.001318       0.892519      0.004650
#     1        0.909256       0.002052       0.902780      0.005053
#     2        0.913621       0.002205       0.906834      0.004423
#     3        0.918600       0.001092       0.910779      0.005221
#     4        0.922251       0.001818       0.914193      0.004422

# %%

# %%
## Limits to cross-validation testing

# Print the first five rows of the CV results data frame
print(cv_results_big.head())

# Calculate the mean of the test AUC scores
print(np.mean(cv_results_big['test-auc-mean']).round(2))

# Plot the test AUC scores for each iteration
plt.plot(cv_results_big['test-auc-mean'])
plt.title('Test AUC Score Over 600 Iterations')
plt.xlabel('Iteration Number')
plt.ylabel('Test AUC Score')
plt.show()

# %%

# %%
## Cross-validation scoring

# Create a gradient boosted tree model using two hyperparameters
gbt = xgb.XGBClassifier(learning_rate = .1, max_depth = 7)

# Calculate the cross validation scores for 4 folds
cv_scores = cross_val_score(gbt, X_train, np.ravel(y_train), cv = 4)

# Print the cross validation scores
print(cv_scores)

# Print the average accuracy and standard deviation of the scores
print("Average accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(),
                                               cv_scores.std() * 2))

# <script.py> output:
#     [0.94095023 0.93369541 0.93186962 0.92462653]
#     Average accuracy: 0.93 (+/- 0.01)

# %%

# %%
## Undersampling training data

# Create data sets for defaults and non-defaults
nondefaults = X_y_train[X_y_train['loan_status'] == 0]
defaults = X_y_train[X_y_train['loan_status'] == 1]

# Undersample the non-defaults
nondefaults_under = nondefaults.sample(len(defaults))

# Concatenate the undersampled nondefaults with defaults
X_y_train_under = pd.concat([nondefaults_under.reset_index(drop = True),
                             defaults.reset_index(drop = True)], axis = 0)

# Print the value counts for loan status
print(X_y_train_under['loan_status'].value_counts())


# <script.py> output:
#     1    3877
#     0    3877
#     Name: loan_status, dtype: int64

# %%

# %%
## Undersampled tree performance

# Check the classification reports
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, gbt_preds, target_names=target_names))
print(classification_report(y_test, gbt2_preds, target_names=target_names))

# <script.py> output:
#                   precision    recall  f1-score   support
#     
#      Non-Default       0.93      0.99      0.96      9198
#          Default       0.95      0.73      0.83      2586
#     
#        micro avg       0.93      0.93      0.93     11784
#        macro avg       0.94      0.86      0.89     11784
#     weighted avg       0.93      0.93      0.93     11784
    
#                   precision    recall  f1-score   support
#     
#      Non-Default       0.95      0.91      0.93      9198
#          Default       0.72      0.84      0.77      2586
    
#        micro avg       0.89      0.89      0.89     11784
#        macro avg       0.83      0.87      0.85     11784
#     weighted avg       0.90      0.89      0.89     11784
______________________________________________________

# Print the confusion matrix for both old and new models
print(confusion_matrix(y_test, gbt_preds))
print(confusion_matrix(y_test, gbt2_preds))

# <script.py> output:
#     0.8613405315086655
#     0.870884117348218
______________________________________________________

# Print and compare the AUC scores of the old and new models
print(roc_auc_score(y_test, gbt_preds))
print(roc_auc_score(y_test, gbt2_preds))

# <script.py> output:
#     0.8613405315086655
#     0.870884117348218

# %%

# %%
## Model Evaluation and Implementation

# Print the logistic regression classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, preds_df_lr['loan_status'], target_names=target_names))

# Print the gradient boosted tree classification report
print(classification_report(y_test, preds_df_gbt['loan_status'], target_names=target_names))

# Print the default F-1 scores for the logistic regression
print(precision_recall_fscore_support(y_test, preds_df_lr['loan_status'], average = 'macro')[2])

# Print the default F-1 scores for the gradient boosted tree
print(precision_recall_fscore_support(y_test, preds_df_gbt['loan_status'], average = 'macro')[2])

________________________________________________________________
# <script.py> output:
#                   precision    recall  f1-score   support
#     
#      Non-Default       0.86      0.92      0.89      9198
#          Default       0.62      0.46      0.53      2586
    
#        micro avg       0.82      0.82      0.82     11784
#        macro avg       0.74      0.69      0.71     11784
#     weighted avg       0.81      0.82      0.81     11784
    
#                   precision    recall  f1-score   support
    
#      Non-Default       0.93      0.99      0.96      9198
#          Default       0.94      0.73      0.82      2586
    
#        micro avg       0.93      0.93      0.93     11784
#        macro avg       0.93      0.86      0.89     11784
#     weighted avg       0.93      0.93      0.93     11784
    
#     0.7108943782814463
#    0.8909014142736051

# %%

# %% [raw]
# 4 - Model Evaluation and Implementation

# %%

# %%
## Comparing with ROCs

# ROC chart components
fallout_lr, sensitivity_lr, thresholds_lr = roc_curve(y_test, clf_logistic_preds)
fallout_gbt, sensitivity_gbt, thresholds_gbt = roc_curve(y_test, clf_logistic_preds)

# ROC Chart with both
plt.plot(fallout_lr, sensitivity_lr, color = 'blue', label='%s' % 'Logistic Regression')
plt.plot(fallout_gbt, sensitivity_gbt, color = 'green', label='%s' % 'GBT')
plt.plot([0, 1], [0, 1], linestyle='--', label='%s' % 'Random Prediction')
plt.title("ROC Chart for LR and GBT on the Probability of Default")
plt.xlabel('Fall-out')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()

________________________________________________________________

# Print the logistic regression AUC with formatting
print("Logistic Regression AUC Score: %0.2f" % roc_auc_score(y_test, clf_logistic_preds))

# Print the gradient boosted tree AUC with formatting
print("Gradient Boosted Tree AUC Score: %0.2f" % roc_auc_score(y_test, clf_gbt_preds))

# <script.py> output:
#     Logistic Regression AUC Score: 0.76
#     Gradient Boosted Tree AUC Score: 0.94

# %%

# %%
## Calibration curves

# Add the calibration curve for the gradient boosted tree
plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')    
plt.plot(mean_pred_val_lr, frac_of_pos_lr,
         's-', label='%s' % 'Logistic Regression')
plt.plot(mean_pred_val_gbt, frac_of_pos_gbt,
         's-', label='%s' % "Gradient Boosted tree")
plt.ylabel('Fraction of positives')
plt.xlabel('Average Predicted Probability')
plt.legend()
plt.title('Calibration Curve')
plt.show()

# %%

# %%
## Acceptance rates

# Check the statistics of the probabilities of default
print(test_pred_df['prob_default'].describe())

# Calculate the threshold for a 85% acceptance rate
threshold_85 = np.quantile(test_pred_df['prob_default'], .85)

# Apply acceptance rate threshold
test_pred_df['pred_loan_status'] = test_pred_df['prob_default'].apply(lambda x: 1 if x > threshold_85 else 0)

# Print the counts of loan status after the threshold
print(test_pred_df['pred_loan_status'].count())

# %% [raw]
# <script.py> output:
#     count    11784.000000
#     mean         0.216866
#     std          0.333038
#     min          0.000354
#     25%          0.022246
#     50%          0.065633
#     75%          0.177804
#     max          0.999557
#     Name: prob_default, dtype: float64
#     0    10016
#     1     1768
#     Name: pred_loan_status, dtype: int64
#
# In the results of .describe() do you see how it's not until 75% that you start to see double-digit numbers? That's because the majority of our test set is non-default loans. Next let's look at how the acceptance rate and threshold split up the data.

# %%

# %%
## Visualizing quantiles of acceptance

# Plot the predicted probabilities of default
plt.hist(clf_gbt_preds, color = 'blue', bins = 40)

# Calculate the threshold with quantile
threshold = np.quantile(clf_gbt_preds, .85)

# Add a reference line to the plot for the threshold
plt.axvline(x =threshold, color = 'red')
plt.show()

# %%

# %%
## Bad rates

# Print the top 5 rows of the new data frame
print(test_pred_df.head())

# Create a subset of only accepted loans
accepted_loans = test_pred_df[test_pred_df['pred_loan_status'] == 0]

# Calculate the bad rate
print(np.sum(accepted_loans['true_loan_status']) / accepted_loans['true_loan_status'].count())

# <script.py> output:
#        true_loan_status  prob_default  pred_loan_status
#     0                 1      0.982387                 1
#     1                 1      0.975163                 1
#     2                 0      0.003474                 0
#     3                 0      0.005457                 0
#     4                 1      0.119876                 0


#     0.08256789137380191

___________________________________________________________________

# This bad rate doesn't look half bad! The bad rate with the threshold 
# set by the 85% quantile() is about 8%. This means that of all the 
# loans we've decided to accept from the test set, only 8% were 
# actual defaults! If we accepted all loans, the percentage 
# of defaults would be around 22%.

# %%

# %%
## Acceptance rate impact

# Print the statistics of the loan amount column
print(test_pred_df['loan_amnt'].describe())

# Store the average loan amount
avg_loan = np.mean(test_pred_df['loan_amnt'])

# Set the formatting for currency, and print the cross tab
pd.options.display.float_format = '${:,.2f}'.format
print(pd.crosstab(test_pred_df['true_loan_status'],
                 test_pred_df['pred_loan_status_15']).apply(lambda x: x * avg_loan, axis = 0))

_______________________________________________________________

# <script.py> output:
#     count    11784.000000
#     mean      9556.283944
#     std       6238.005674
#     min        500.000000
#     25%       5000.000000
#     50%       8000.000000
#     75%      12000.000000
#     max      35000.000000
#     Name: loan_amnt, dtype: float64


#     pred_loan_status_15              0              1
#     true_loan_status                                 
#     0                   $87,812,693.16     $86,006.56
#     1                    $7,903,046.82 $16,809,503.46


# %%

# %%
## Making the strategy table

accept_rates = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55,
                0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

# Create lists to store thresholds and bad rates
thresholds = []
bad_rates = []

# Populate the arrays for the strategy table with a for loop
for rate in accept_rates:
    
    # Calculate the threshold for the acceptance rate
    thresh = np.quantile(preds_df_gbt['prob_default'], \
                         rate).round(3)
    
    # Add the threshold value to the list of thresholds
    thresholds.append(np.quantile(preds_df_gbt['prob_default'],\
                                  rate).round(3))
    
    # Reassign the loan_status value using the threshold
    test_pred_df['pred_loan_status'] = test_pred_df['prob_default']\
                                        .apply(lambda x: 1 if x > thresh else 0)
    
    # Create a set of accepted loans using this acceptance rate
    accepted_loans = test_pred_df[test_pred_df['pred_loan_status'] == 0]
    
    # Calculate and append the bad rate using the acceptance rate
    bad_rates.append(np.sum((accepted_loans['true_loan_status']) /\
                            len(accepted_loans['true_loan_status'])).round(3))
    
# Create a data frame of the strategy table
strat_df = pd.DataFrame(zip(accept_rates, thresholds, bad_rates),
                        columns = ['Acceptance Rate','Threshold','Bad Rate'])

# Print the entire table
print(strat_df)

# %% [raw]
# <script.py> output:
#         Acceptance Rate  Threshold  Bad Rate
#     0              1.00      1.000     0.219
#     1              0.95      0.992     0.179
#     2              0.90      0.976     0.132
#     3              0.85      0.804     0.083
#     4              0.80      0.254     0.061
#     5              0.75      0.178     0.052
#     6              0.70      0.138     0.043
#     7              0.65      0.111     0.036
#     8              0.60      0.093     0.030
#     9              0.55      0.078     0.027
#     10             0.50      0.066     0.023
#     11             0.45      0.055     0.020
#     12             0.40      0.045     0.017
#     13             0.35      0.037     0.014
#     14             0.30      0.030     0.010
#     15             0.25      0.022     0.008
#     16             0.20      0.015     0.005
#     17             0.15      0.008     0.001
#     18             0.10      0.004     0.000
#     19             0.05      0.002     0.000

# %%

# %%
## Visualizing the strategy

# Visualize the distributions in the strategy table with a boxplot
strat_df.boxplot()
plt.show()
______________________________________

# Plot the strategy curve
plt.plot(strat_df['Acceptance Rate'], 
         strat_df['Bad Rate'])
plt.xlabel('Acceptance Rate')
plt.ylabel('Bad Rate')
plt.title('Acceptance and Bad Rates')
plt.axes().yaxis.grid()
plt.axes().xaxis.grid()
plt.show()

# %%

# %% [raw]
#
# Column	------------ Description
# Num Accepted Loans	 The number of accepted loans based on the threshold
# Avg Loan Amnt ------ The average loan amount of the entire test set
# Estimated value ---- The estimated net value of non-defaults minus defaults

# %%
## Estimated value profiling

# Print the row with the max estimated value
print(strat_df.loc[strat_df['Estimated Value'] == np.max(strat_df['Estimated Value'])])

# <script.py> output:
#        Acceptance Rate  Threshold  Bad Rate  Num Accepted Loans  Avg Loan Amnt  Estimated Value
#     3             0.85      0.804     0.083                9390        9556.28      74837713.31

# %%

# %%
## Total expected loss

# Print the first five rows of the data frame
print(test_pred_df.head())

# Calculate the bank's expected loss and assign it to a new column
test_pred_df['expected_loss'] = test_pred_df['prob_default'] * test_pred_df['loss_given_default'] * test_pred_df['loan_amnt']

# Calculate the total expected loss to two decimal places
tot_exp_loss = round(np.sum(test_pred_df['expected_loss']),2)

# Print the total expected loss
print('Total expected loss: ', '${:,.2f}'.format(tot_exp_loss))


# <script.py> output:
#        true_loan_status  prob_default  loan_amnt  loss_given_default
#     0                 1      0.982387      15000                 1.0
#     1                 1      0.975163      11200                 1.0
#     2                 0      0.003474      15000                 1.0
#     3                 0      0.005457      10800                 1.0
#     4                 1      0.119876       3000                 1.0

#     Total expected loss:  $27,084,153.38

# %%

# %%
