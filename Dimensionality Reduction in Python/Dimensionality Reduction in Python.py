# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
## Removing features without variance

# Leave this list as is
number_cols = ['HP', 'Attack', 'Defense']

# Remove the feature without variance from this list
non_number_cols = ['Name', 'Type', 'Legendary']

# Create a new dataframe by subselecting the chosen features
df_selected = pokemon_df[number_cols + non_number_cols]

# Prints the first 5 lines of the new dataframe
print(df_selected.head())
# -



# +
## Fitting t-SNE to the ANSUR data

# Non-numerical columns in the dataset
non_numeric = ['Branch', 'Gender', 'Component']

# Drop the non-numerical columns from df
df_numeric = df.drop(non_numeric, axis=1)

# Create a t-SNE model with learning rate 50
m = TSNE(learning_rate=50)

# Fit and transform the t-SNE model on the numeric dataset
tsne_features = m.fit_transform(df_numeric)
print(tsne_features)
# -



# +
# Color the points according to Army Component
sns.scatterplot(x="x", y="y", hue='Component', data=df)

# Show the plot
plt.show()

# Color the points according to Army Component
sns.scatterplot(x="x", y="y", hue='Branch', data=df)

# Show the plot
plt.show()

# Color the points according to Army Component
sns.scatterplot(x="x", y="y", hue='Gender', data=df)

# Show the plot
plt.show()
# -



# +
## Module 2

# +
## Train - test split

# Import train_test_split()
from sklearn.model_selection import train_test_split

# Select the Gender column as the feature to be predicted (y)
y = ansur_df['Gender']

# Remove the Gender column to create the training data
X = ansur_df.drop('Gender', axis=1)

# Perform a 70% train and 30% test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

print("{} rows in test set vs. {} in training set. {} Features.".format(X_test.shape[0], X_train.shape[0], X_test.shape[1]))
# -



# +
## Fitting and testing the model

# Import SVC from sklearn.svm and accuracy_score from sklearn.metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create an instance of the Support Vector Classification class
svc = SVC()

# Fit the model to the data
svc.fit(X_train, y_train)

# Calculate accuracy scores on both train and test data
accuracy_train = accuracy_score(y_train, svc.predict(X_train))
accuracy_test = accuracy_score(y_test, svc.predict(X_test))

print("{0:.1%} accuracy on test set vs. {1:.1%} on training set".format(accuracy_test, accuracy_train))
# -



# +
## Features with low variance

from sklearn.feature_selection import VarianceThreshold

# Create a VarianceThreshold feature selector
sel = VarianceThreshold(threshold=0.001)

# Fit the selector to normalized head_df
sel.fit(head_df / head_df.mean())

# Create a boolean mask
mask = sel.get_support()

# Apply the mask to create a reduced dataframe
reduced_df = head_df.loc[:, mask]

print("Dimensionality reduced from {} to {}.".format(head_df.shape[1], reduced_df.shape[1]))
# -



# +
## Removing features with many missing values

# Create a boolean mask on whether each feature less than 50% missing values.
mask = school_df.isna().sum() / len(school_df) < 0.5

# Create a reduced dataset by applying the mask
reduced_df = school_df.loc[:, mask]

print(school_df.shape)
print(reduced_df.shape)
# -



# +
## Visualizing the correlation matrix

# Create the correlation matrix
corr = ansur_df.corr()

# Generate a mask for the upper triangle 
mask = np.triu(np.ones_like(corr, dtype=bool))

# Add the mask to the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
plt.show()
# -



# +
## Filtering out highly correlated features


# Calculate the correlation matrix and take the absolute value
corr_matrix = ansur_df.corr().abs()

# Create a True/False mask and apply it
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask)

# List column names of highly correlated features (r > 0.95)
to_drop = [c for c in tri_df.columns if any(tri_df[c] >  0.95)]

# Drop the features in the to_drop list
reduced_df = ansur_df.drop(to_drop, axis=1)

print("The reduced dataframe has {} columns.".format(reduced_df.shape[1]))

#The original dataframe has 99 columns.

# <script.py> output:
#    The reduced dataframe has 88 columns.
# -

## Nuclear energy and pool drownings



# +
## Module 3

# +
## Building a diabetes classifier

# Scale the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Perform a 25-75% train test split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=.25, random_state=0)

# Create the logistic regression model and fit it to the data
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Calculate the accuracy on the test set
acc = accuracy_score(y_test, lr.predict(X_test))
print("{0:.1%} accuracy on test set.".format(acc)) 
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))
# -



# +
# Remove the feature with the lowest model coefficient
X = diabetes_df[['pregnant', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi', 'family', 'age']]

# Scales the features and performs a 25-75% train test split
X_std = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.25, random_state=0)

# Creates the logistic regression model and fits it to the data
lr.fit(X_train, y_train)

# Calculates the accuracy on the test set and prints coefficients
acc = accuracy_score(y_test, lr.predict(X_test))
print("{0:.1%} accuracy on test set.".format(acc)) 
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))

#######

#######

#######

# Only keep the feature with the highest coefficient
X = diabetes_df[['glucose']]

# Scales the features and performs a 25-75% train test split
X_std = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.25, random_state=0)

# Creates the logistic regression model and fits it to the data
lr.fit(X_train, y_train)

# Calculates the accuracy on the test set and prints coefficients
acc = accuracy_score(y_test, lr.predict(X_test))
print("{0:.1%} accuracy on test set.".format(acc)) 
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))
# -



# +
# Automatic Recursive Feature Elimination
# Now let's automate this recursive process. Wrap a Recursive 
# Feature Eliminator (RFE) around our logistic regression estimator 
# and pass it the desired number of features.


# Create the RFE with a LogisticRegression estimator and 3 features to select
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=3, verbose=1)

# Fits the eliminator to the data
rfe.fit(X_train, y_train)

# Print the features and their ranking (high = dropped early on)
print(dict(zip(X.columns, rfe.ranking_)))

# Print the features that are not eliminated
print(X.columns[rfe.support_])

# Calculates the test set accuracy
acc = accuracy_score(y_test, rfe.predict(X_test))
print("{0:.1%} accuracy on test set.".format(acc)) 
# -



# +
#### Tree-based feature selection ####

## Building a random forest model

# Perform a 75% training and 25% test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

# Fit the random forest model to the training data
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

# Calculate the accuracy
acc = accuracy_score(y_test, rf.predict(X_test))

# Print the importances per feature
print(dict(zip(X.columns, rf.feature_importances_.round(2))))

# Print accuracy
print("{0:.1%} accuracy on test set.".format(acc))
# -



# +
## Random forest for feature selection

# Create a mask for features importances above the threshold
mask = rf.feature_importances_ > 0.15

# Apply the mask to the feature dataset X
reduced_X = X.loc[:, mask]

# prints out the selected column names
print(reduced_X.columns)
# -



# +
## Recursive Feature Elimination with random forests

# Wrap the feature eliminator around the random forest model
rfe = RFE(estimator=RandomForestClassifier(random_state=0), n_features_to_select=2, verbose=1)

## Second part 

# Set the feature eliminator to remove 2 features on each step
rfe = RFE(estimator=RandomForestClassifier(random_state=0), n_features_to_select=2, step=2, verbose=1)

# Fit the model to the training data
rfe.fit(X_train, y_train)

# Create a mask
mask = rfe.support_

# Apply the mask to the feature dataset X and print the result
reduced_X = X.loc[:, mask]
print(reduced_X.columns)
# -



# +
## Creating a LASSO regressor

scaler = StandardScaler()

# Fit the scaler to X and transform the input features
X_std = scaler.fit_transform(X)

# Set the test size to 30% to get a 70-30% train test split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=.3, random_state=0)

# Create the Lasso model and fit it to the training data
la = Lasso()
la.fit(X_train, y_train)
# -



# +
## Lasso model results

# Calculate the coefficient of determination (R squared) on the test set
r_squared = la.score(X_test, y_test)
print("The model can predict {0:.1%} of the variance in the test set.".format(r_squared))

# Sum the number of features for which the coefficients are 0
n_ignored = sum(la.coef_ == 0)
print("The model has ignored {} out of {} features.".format(n_ignored, len(la.coef_)))

# +
## Adjusting the regularization strength

# Find the right alpha value
la = Lasso(alpha=0.1, random_state=0)

# Fits the model and calculates performance stats
la.fit(X_train, y_train)
r_squared = la.score(X_test, y_test)
n_ignored_features = sum(la.coef_==0)

# Print peformance stats 
print("The model can predict {0:.1%} of the variance in the test set.".format(r_squared))
print("{} out of {} features were ignored.".format(n_ignored_features, len(la.coef_)))

# +
# <script.py> output:
#    The model can predict 98.3% of the variance in the test set.
#    64 out of 91 features were ignored.
    
# Wow! We this more appropriate regularization strength we can predict 98% 
# of the variance in the BMI value while ignoring 2/3 of the features.
# -



# +
### Combining feature selectors

## Creating a LassoCV regressor

from sklearn.linear_model import LassoCV

# Create and fit the LassoCV model on the training set
lcv = LassoCV()
lcv.fit(X_train, y_train)
print('Optimal alpha = {0:.3f}'.format(lcv.alpha_))

# Calculate R squared on the test set
r_squared = lcv.score(X_test, y_test)
print('The model explains {0:.1%} of the test set variance'.format(r_squared))

# Create a mask for coefficients not equal to zero
lcv_mask = lcv.coef_ != 0
print('{} features out of {} selected'.format(sum(lcv_mask), len(lcv_mask)))


# <script.py> output:
#    Optimal alpha = 0.085
#    The model explains 88.3% of the test set variance
#    27 features out of 32 selected

# +
## Ensemble models for extra votes

from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor

# Select 10 features with RFE on a GradientBoostingRegressor, drop 3 features on each step
rfe_gb = RFE(estimator=GradientBoostingRegressor(), 
             n_features_to_select=10, step=3, verbose=1)
rfe_gb.fit(X_train, y_train)

# Calculate the R squared on the test set
r_squared = rfe_gb.score(X_test, y_test)
print('The model can explain {0:.1%} of the variance in the test set'.format(r_squared))

# Assign the support array to gb_mask
gb_mask = rfe_gb.support_

# + active=""
# <script.py> output:
#     Fitting estimator with 32 features.
#     Fitting estimator with 29 features.
#     Fitting estimator with 26 features.
#     Fitting estimator with 23 features.
#     Fitting estimator with 20 features.
#     Fitting estimator with 17 features.
#     Fitting estimator with 14 features.
#     Fitting estimator with 11 features.
#     The model can explain 85.6% of the variance in the test set

# +
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

# Select 10 features with RFE on a RandomForestRegressor, drop 3 features on each step
rfe_rf = RFE(estimator=RandomForestRegressor(), 
             n_features_to_select=10, step=3, verbose=1)
rfe_rf.fit(X_train, y_train)

# Calculate the R squared on the test set
r_squared = rfe_rf.score(X_test, y_test)
print('The model can explain {0:.1%} of the variance in the test set'.format(r_squared))

# Assign the support array to gb_mask
rf_mask = rfe_rf.support_

# + active=""
# from sklearn.feature_selection import RFE
# from sklearn.ensemble import RandomForestRegressor
#
# # Select 10 features with RFE on a RandomForestRegressor, drop 3 features on each step
# rfe_rf = RFE(estimator=RandomForestRegressor(), 
#              n_features_to_select=10, step=3, verbose=1)
# rfe_rf.fit(X_train, y_train)
#
# # Calculate the R squared on the test set
# r_squared = rfe_rf.score(X_test, y_test)
# print('The model can explain {0:.1%} of the variance in the test set'.format(r_squared))
#
# # Assign the support array to gb_mask
# rf_mask = rfe_rf.support_
# -



# +
## Combining 3 feature selectors

# Sum the votes of the three models
votes = np.sum([lcv_mask, rf_mask, gb_mask], axis=0)

# Create a mask for features selected by all 3 models
meta_mask = votes >= 3

# Apply the dimensionality reduction on X
X_reduced = X.loc[:, meta_mask]

# Plug the reduced dataset into a linear regression pipeline
X_std = scaler.fit_transform(X_reduced)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=0)
lm.fit(X_train, y_train)
r_squared = lm.score(X_test, y_test)
print('The model can explain {0:.1%} of the variance \
       in the test set using {1:} features.'
      .format(r_squared, len(lm.coef_)))

# +
# <script.py> output:
#    The model can explain 86.8% of the variance in the test set using 7 features.
# -



# +
## Module 4
# -



# +
## Manual feature extraction I

# Calculate the price from the quantity sold and revenue
sales_df['price'] = sales_df['revenue'] / sales_df['quantity']

# Drop the quantity and revenue features
sales_df.drop(['revenue','quantity'], axis=1, inplace=True)

print(sales_df.head())
# -



# +
# Manual feature extraction II

# Calculate the mean height
height_df['height'] = (height_df['height_1'] + height_df['height_2'] + height_df['height_3']) / 3

# Drop the 3 original height features
height_df.drop(['height_1', 'height_2', 'height_3'], axis=1, inplace=True)

print(height_df.head())
# -



# +
## Calculating Principal Components

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create the scaler
scaler = StandardScaler()
ansur_std = scaler.fit_transform(ansur_df)

# Create the PCA instance and fit and transform the data with pca
pca = PCA()
pc = pca.fit_transform(ansur_std)
pc_df = pd.DataFrame(pc, columns=['PC 1', 'PC 2', 'PC 3', 'PC 4'])

# Create a pairplot of the principal component dataframe
sns.pairplot(pc_df)
plt.show()
# -



# +
## PCA on a larger dataset

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scale the data
scaler = StandardScaler()
ansur_std = scaler.fit_transform(ansur_df)

# Apply PCA
pca = PCA()
pca.fit(ansur_std)
# -

# Print the cumulative sum of the explained variance ratio
print(pca.explained_variance_ratio_.cumsum())



# +
## Understanding the components

# Build the pipeline
pipe = Pipeline([('scaler', StandardScaler()),
        		 ('reducer', PCA(n_components=2))])

# Fit it to the dataset and extract the component vectors
pipe.fit(poke_df)
vectors = pipe.steps[1][1].components_.round(2)

# Print feature effects
print('PC 1 effects = ' + str(dict(zip(poke_df.columns, vectors[0]))))
print('PC 2 effects = ' + str(dict(zip(poke_df.columns, vectors[1]))))
# -



# +
## PCA in a model pipeline

# Build the pipeline
pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reducer', PCA(n_components=3)),
        ('classifier', RandomForestClassifier(random_state=0))])

# Fit the pipeline to the training data
pipe.fit(X_train, y_train)

# Score the accuracy on the test set
accuracy = pipe.score(X_test, y_test)

# Prints the explained variance ratio and accuracy
print(pipe.steps[1][1].explained_variance_ratio_)
print('{0:.1%} test set accuracy'.format(accuracy))
# -



# +
## Selecting the proportion of variance to keep

# Let PCA select 90% of the variance
pipe = Pipeline([('scaler', StandardScaler()),
        		 ('reducer', PCA(n_components=.9))])

# Fit the pipe to the data
pipe.fit(ansur_df)

print('{} components selected'.format(len(pipe.steps[1][1].components_)))
# -



# +
## Choosing the number of components

# Pipeline a scaler and pca selecting 10 components
pipe = Pipeline([('scaler', StandardScaler()),
        		 ('reducer', PCA(n_components=10))])

# Fit the pipe to the data
pipe.fit(ansur_df)

# Plot the explained variance ratio
plt.plot(pipe.steps[1][1].explained_variance_ratio_)

plt.xlabel('Principal component index')
plt.ylabel('Explained variance ratio')
plt.show()
# -



# +
## PCA for image compression

# Transform the input data to principal components
pc = pipe.transform(X_test)

# Prints the number of features per dataset
print("X_test has {} features".format(X_test.shape[1]))
print("pc has {} features".format(pc.shape[1]))



# +
# Transform the input data to principal components
pc = pipe.transform(X_test)

# Inverse transform the components to original feature space
X_rebuilt = pipe.inverse_transform(pc)

# Prints the number of features
print("X_rebuilt has {} features".format(X_rebuilt.shape[1]))

# Plot the reconstructed data
plot_digits(X_rebuilt)
# -












