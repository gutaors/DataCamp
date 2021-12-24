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

# # Narrando o aprendizado
# Este caderno começa como todos os outros, com os imports mais básicos e jogando o matplotlib pra inline
# importa o csv, 
#     qual a biblioteca do comando que lê csv?
#     qual o comando?
#     qual a sintaxe para passar o nome do arquivo?
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline




import inspect
#source = inspect.getsource(function)
#print(source)

# ## Combining Multiple Models
#
# Como ele combina? 
# Qual o passo a passo? 
# Tem pré processamento?
# tente escrever a célula e confira com a do cara para validar, lembre de no começo escrever #MinhaCélula
#

app = pd.read_csv('googleplaystore.csv')

 #MinhaCélula
app.head(3)


# # Comando para mostrar quantidade de null por variável, dica do Christian

app.isna().sum()

# ### Aqui o malaco fez o seguinte, pegou o type que falava se era gratuito ou não e transformou no tipo Paid que tem 0 ou 1
# É interessante notar que o get_dummies é um tipo do pandas e ele já tem este tipo Paid, tentei mudar para Pagou e deu erro

dummy = pd.get_dummies(app['Type'])
app = app.drop('Type', axis = 1)
app = pd.concat([app, dummy['Paid']], axis = 1)



# + active=""
# #### abaixo ele faz diferente, troca a palavra Teen por 10 e depois roda uma expressão que transforma o campo Content Rating em números, bota num float e arredonda o float
# -

app['Content Rating'] = app['Content Rating'].replace('Teen', '10')
app['Content Rating'] = app['Content Rating'].str.extract(r'(\d+)').astype(float)
app['Content Rating'] = app['Content Rating'].fillna(0)

# #### campo Size, converte em número também ver qual a lógica, já sei que quando o campo está preenchido com Varies whth device ele arranca fora

app['Size'] = app['Size'].apply(lambda x: str(float(x.replace('k', '')) / 1000) if 'k' in x else x)
app['Size'] = app['Size'].replace('Varies with device', np.nan)

# #### Várias operações aqui, lindo este código aqui
# price = trocou Everyone por 0
#
# trocou todos free por 0
#
# definiu alguns caracteres em uma lista
#
#
# definiu alguns nomes de colunas em outra lista
#
# fez um luoop percorrendo as colunas
#
#
#     dentro deste loop percorreu os caracteres
#     
#        para cada coluna destas no dataframe, trocou o char por ''
#        

app['Price'] = app['Price'].replace('Everyone', '0')
app = app.replace('Free', 0)
chars = ['+', '$', 'M', ',']
cols = ['Reviews','Size', 'Installs', 'Price']
for col in cols:
    for char in chars:
        app[col] = app[col].str.replace(char, '')

# Mudou o tipo destas colunas que são as mesmas que acabou de tratar
# Em seguida rodou dropna para remover os null

app[['Reviews', 'Size', 'Installs', 'Price']].astype(float)
app = app.dropna()

# #### Separou as colunas X e y, lembrando que ao contrário da intuição, y é o alvo

X = app[['Reviews', 'Size', 'Installs', 'Paid', 'Price', 'Content Rating']]
y = app['Rating']

# print(y)


# ### Predicting the rating of an app

# Separa em X e y, qual função que usa pra isto? esta função pertence a quem?
# Rodou um regressor de Árvore de Decisão, não é random forest, é um regressor porque estou tentando achar uma estimativa de  valor, na dúvida veja o print(y) acima para verificar o que vou encontrar

# +
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Split into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Instantiate the regressor
reg_dt = DecisionTreeRegressor(min_samples_leaf = 3, min_samples_split = 9, random_state=500)

# Fit to the training set
reg_dt.fit(X_train, y_train)
# -

# ### Choosing the best model

# #### aqui o esquema é outro, vamos abrir outro csv e fazer outras coisas divertidas
# #### parece que o bixinho já está bem preparado pois não tem pré processamento aqui. Já abre, separa colunas e mente bronca no split

pkm = pd.read_csv('Pokemon.csv', index_col = 0)
X = pkm[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']]
y = pkm['Legendary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
pkm.head()

# ### Pronto, agora começa a evoluir, vamos abrir uma porrada de biblioteca

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

# #### fez fit em três modelos, regressão logística, classificador árvore de decisão e Classificador KNN

clf_lr = LogisticRegression(solver = 'liblinear').fit(X_train, y_train)
clf_dt = DecisionTreeClassifier().fit(X_train, y_train)
clf_knn = KNeighborsClassifier().fit(X_train, y_train)

# #### avalia os modelos da seguinte forma
# cria uma variável pred para cada algoritmo
# Como? as predições eu vou jogar em uma variável pred_.. cada predict que fiz, passo o modelo que fiz fit ali em cima .predict(X_test) faço a previsão para o X_test
#
# avalio a qualidade verificando contra o y_test contra o pred que criei logo acima, jogo o resultado em uma lista score
#
# printo os scores

# +
# Make the invidual predictions
pred_lr = clf_lr.predict(X_test)
pred_dt = clf_dt.predict(X_test)
pred_knn = clf_knn.predict(X_test)

# Evaluate the performance of each model
score_lr = f1_score(y_test, pred_lr)
score_dt = f1_score(y_test, pred_dt)
score_knn = f1_score(y_test, pred_knn)

# Print the scores
print(score_lr)
print(score_dt)
print(score_knn)
# -

# ### Assembling your first ensemble

# #### montando seu primeiro conjunto
# instancia cada modelo
#
# no knn o parâmetro é número de vizinhos
#
# no LogisticRegression tem o solver que é linear
#
# e no Classificador Árvore de decisão tem leaf, split e random state
#
# dá um nome pra cada e depois faz um VotingClassifier que vai chamar os três gerando um X_train e yTrain, 
#
# Lembra daquele lance de ver as questões que cada aluno resolve bem para colocá-los para resolver a prova? Está aí
# Que parãmetros passaos para o VotingClassifier? estimators =['abrev', nome_instancia]
# Faz vote.fit(X_treino e y_treino

# +
# Instantiate the individual models
clf_knn = KNeighborsClassifier(n_neighbors = 5)
clf_lr = LogisticRegression(class_weight = 'balanced', solver = 'liblinear')
clf_dt = DecisionTreeClassifier(min_samples_leaf=3, min_samples_split=9, random_state=500)

# Create and fit the voting classifier
clf_vote = VotingClassifier(
    estimators=[('knn', clf_knn), ('lr', clf_lr), ('dt', clf_dt)]
)
clf_vote.fit(X_train, y_train)
# -

# ### Evaluating your ensemble

# #### O modelo do Ensemble é tratado como um modelo qualquer, acima fizemos o fit nele, agora faremos o predict com X_test, 
#
# depois faremos o score comparando pred_vote (nossas previsões) com o y_test e depois vamos fazer o print
#
# fazemos o report com o classification_report(y_test, pred_vote) ou seja, um relatório de classificação que compara o y (alvo) e o previsto

# +
# Calculate the predictions using the voting classifier
pred_vote = clf_vote.predict(X_test)

# Calculate the F1-Score of the voting classifier
score_vote = f1_score(y_test, pred_vote)
print('F1-Score: {:.3f}'.format(score_vote))

# Calculate the classification report
report = classification_report(y_test, pred_vote)
print(report)
# -

# ### Predicting GoT deaths

# #### leu o dataset, definiu quais features vão pro X, qual é o alvo (y)
# fez o split no dataset Game of Thrones, massa demais
#

got = pd.read_csv('character-predictions.csv').fillna(0)
X = got[['male', 'book1', 'book2', 'book3', 'book4', 'book5', 'isAliveMother', 'isAliveFather', 'isAliveHeir', 'isAliveSpouse', 'isMarried', 'isNoble', 'age', 'numDeadRelations', 'boolDeadRelations','isPopular', 'popularity']]
y = got['isAlive']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
got.head()


# ### Faz um modelo de cada coisa, LogisticRegression, DecisionTreeClassifier

# +
# Build the individual models
clf_lr = LogisticRegression(class_weight='balanced', solver = 'liblinear')
clf_dt = DecisionTreeClassifier(min_samples_leaf=3, min_samples_split=9, random_state=500)
clf_svm = SVC(probability=True, class_weight='balanced', random_state=500, gamma = 'auto')

# List of (string, estimator) tuples
estimators = [('lr', clf_lr), ('dt', clf_dt), ('svm', clf_svm)]

# Build and fit an averaging classifier
clf_avg = VotingClassifier(estimators, voting='soft')
clf_avg.fit(X_train, y_train)

# Evaluate model performance
acc_avg = accuracy_score(y_test,  clf_avg.predict(X_test))
print('Accuracy: {:.2f}'.format(acc_avg))
# -

# ### Soft vs. hard voting
# O que é soft vs hard sob o ponto de vista da implementação?

# +
# List of (string, estimator) tuples
estimators = [('dt', clf_dt), ('lr', clf_lr), ('knn', clf_knn)]

# Build and fit a voting classifier
clf_vote = VotingClassifier(estimators = estimators)
clf_vote.fit(X_train, y_train)

# Build and fit an averaging classifier
clf_avg = VotingClassifier(estimators = estimators, voting = 'soft')
clf_avg.fit(X_train, y_train)

# Evaluate the performance of both models
acc_vote = accuracy_score(y_test, clf_vote.predict(X_test))
acc_avg = accuracy_score(y_test,  clf_avg.predict(X_test))
print('Voting: {:.2f}, Averaging: {:.2f}'.format(acc_vote, acc_avg))
# -

# ## Bagging

# ### Restricted and unrestricted decision trees
# O que são as árvores restritas e irrestritas sob o ponto de vista da implementação??
#

X = pkm[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']]
y = pkm['Legendary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# +
# Build unrestricted decision tree
clf = DecisionTreeClassifier(min_samples_leaf = 3, min_samples_split = 9, random_state = 500)
clf.fit(X_train, y_train)

# Predict the labels
pred = clf.predict(X_test)

# Print the confusion matrix
cm = confusion_matrix(y_test, pred)
print('Confusion matrix:\n', cm)

# Print the F1 score
score = f1_score(y_test, pred)
print('F1-Score: {:.3f}'.format(score))

# +
# Build restricted decision tree
clf = DecisionTreeClassifier(max_depth = 4, max_features=2, random_state=500)
clf.fit(X_train, y_train)

# Predict the labels
pred = clf.predict(X_test)

# Print the confusion matrix
cm = confusion_matrix(y_test, pred)
print('Confusion matrix:\n', cm)

# Print the F1 score
score = f1_score(y_test, pred)
print('F1-Score: {:.3f}'.format(score))
# -

# ### Training with bootstrapping
# Como treina com bootstrapping? que bagaça é esta?

pkm_sample = pkm.sample(640, replace = True)
X = pkm_sample[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']]
y = pkm_sample['Legendary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# +
# Take a sample with replacement
# X_train_sample = X_train.sample(frac = 1.0, replace = True, random_state=42)
# y_train_sample = y_train.loc[X_train_sample.index]


# Build a "weak" Decision Tree classifier
clf = DecisionTreeClassifier(max_depth = 4, max_features = 2, random_state=500)

# Fit the model to the training sample
clf.fit(X_train, y_train)
# -

# ### Bagging: the scikit-learn way
# Lembra do bagging? Qual(is) o(s) comando(s)?
#
#
#
# BAGGING ou BOOTSTRAP AGGREGATION
# Arvores profundas tendem a Overfitar e levam a mais variancia
# na random forest o treino é dividido em amostras aleatorias com reposição, gerando pequenos subsets. chamados de bootstrap samples
# Estes bootstraps são servidos para alimentar árvores de grande profundidade.
# cada árvore destas é treinada separadamente nestas amostras (bootstrap)
# Esta agregação de árvores é chamado Random Forest Ensemble 
# O resultado final é obtido pelo voto da maioria, isto é chamado de Bagging ou Bootstrap Aggregation.
# Como cada árvore pegou um set de dados como amostra de treino, os desvios no dataset de treino original não impacta o resultado final obtido pela agregação de DTs
# Bagging reduz a variância sem alterar o BIAS do ensemble.
#
# OOB Score (Out of the bag)
# Deixamos linhas fora do dataset de treino
#

# +
from sklearn.ensemble import BaggingClassifier
# Instantiate the base model
clf_dt = DecisionTreeClassifier(max_depth = 4)

# Build and train the Bagging classifier
clf_bag = BaggingClassifier(
  clf_dt,
  21,
  random_state=500)
clf_bag.fit(X_train, y_train)

# Predict the labels of the test set
pred = clf_bag.predict(X_test)

# Show the F1-score
print('F1-Score: {:.3f}'.format(f1_score(y_test, pred)))
# -

# ### Checking the out-of-bag score
# OOB, lembra? não ? dá uma lida  na net e depois olha o código

# +
# Build and train the bagging classifier
clf_bag = BaggingClassifier(
  base_estimator=clf_dt,
  n_estimators=21,
  oob_score=True,
  random_state=500)
clf_bag.fit(X_train, y_train)

# Print the out-of-bag score
print('OOB-Score: {:.3f}'.format(clf_bag.oob_score_))

# Evaluate the performance on the test set to compare
pred = clf_bag.predict(X_test)
print('Accuracy: {:.3f}'.format(accuracy_score(y_test, pred)))
# -

# ### A more complex bagging model
#

uci = pd.read_csv('uci-secom.csv', index_col = 'Time').fillna(0)
X = uci.iloc[:, :590]
y = uci['Pass/Fail']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# ### regressão logística balanceada, classificador bagging e OOB, depois matriz de confusão

# +
# Build a balanced logistic regression
clf_lr = LogisticRegression(class_weight = 'balanced', solver = 'liblinear')

# Build and fit a bagging classifier
clf_bag = BaggingClassifier(base_estimator = clf_lr, n_estimators = 10, oob_score = True, max_features = 10, random_state=500)
clf_bag.fit(X_train, y_train)

# Evaluate the accuracy on the test set and show the out-of-bag score
pred = clf_bag.predict(X_test)
print('Accuracy:  {:.2f}'.format(accuracy_score(y_test, pred)))
print('OOB-Score: {:.2f}'.format(clf_bag.oob_score_))

# Print the confusion matrix
print(confusion_matrix(y_test, pred))
# -

# ### Tuning bagging hyperparameters

# +
# Build a balanced logistic regression
clf_base = LogisticRegression(class_weight='balanced', random_state=42, solver = 'liblinear')

# Build and fit a bagging classifier with custom parameters
clf_bag = BaggingClassifier(base_estimator = clf_base, n_estimators = 500, max_features= 10, max_samples = 0.65, bootstrap = False, random_state=500)
clf_bag.fit(X_train, y_train)

# Calculate predictions and evaluate the accuracy on the test set
y_pred = clf_bag.predict(X_test)
print('Accuracy:  {:.2f}'.format(accuracy_score(y_test, y_pred)))

# Print the classification report
print(classification_report(y_test, y_pred))
# -

# ## Boosting

# ### Predicting movie revenue

movie = pd.read_csv('tmdb_5000_movies.csv')
X = movie['budget']
y = movie['revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train.values.reshape(-1, 1))
X_test = scaler.transform(X_test.values.reshape(-1,1))

# +
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Build and fit linear regression model
reg_lm = LinearRegression(normalize = True)
reg_lm.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = reg_lm.predict(X_test)

# Evaluate the performance using the RMSE
rmse = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE: {:.3f}'.format(rmse))
# -

# ### Your first AdaBoost model

movie = pd.read_csv('tmdb_5000_movies.csv')
X = movie[['budget', 'popularity']]
y = movie['revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# +
from sklearn.ensemble import AdaBoostRegressor
# Instantiate a normalized linear regression model
reg_lm = LinearRegression(normalize = True)

# Build and fit an AdaBoost regressor
reg_ada = AdaBoostRegressor(base_estimator = reg_lm,n_estimators = 12,  random_state=500)
reg_ada.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = reg_ada.predict(X_test)

# Evaluate the performance using the RMSE
rmse = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE: {:.3f}'.format(rmse))
# -

# ### Tree-based AdaBoost regression

# +
# Build and fit a tree-based AdaBoost regressor
reg_ada = AdaBoostRegressor(n_estimators = 12, random_state=500)
reg_ada.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = reg_ada.predict(X_test)

# Evaluate the performance using the RMSE
rmse = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE: {:.3f}'.format(rmse))
# -

# ### Making the most of AdaBoost

movie = pd.read_csv('tmdb_5000_movies.csv').dropna()
X = movie[['budget', 'popularity', 'runtime', 'vote_average','vote_count']]
y = movie['revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# +
# Build and fit an AdaBoost regressor
reg_ada = AdaBoostRegressor(n_estimators = 100, learning_rate = 0.01, random_state=500)
reg_ada.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = reg_ada.predict(X_test)

# Evaluate the performance using the RMSE
rmse = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE: {:.3f}'.format(rmse))
# -

# ### Sentiment analysis with GBM

# +
from sklearn.ensemble import GradientBoostingClassifier
# Build and fit a Gradient Boosting classifier
clf_gbm = GradientBoostingClassifier(n_estimators = 5, learning_rate = 0.1, random_state=500)
clf_gbm.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = clf_gbm.predict(X_test)

# Evaluate the performance based on the accuracy
acc = accuracy_score(y_test, pred)
print('Accuracy: {:.3f}'.format(acc))

# Get and show the Confusion Matrix
cm = confusion_matrix(y_test, pred)
print(cm)
# -

# ### Movie revenue prediction with CatBoost

# +
# import catboost as cb
# # Build and fit a CatBoost regressor
# reg_cat = cb.CatBoostRegressor(n_estimators = 5, learning_rate = 0.1, max_depth = 3, random_state=500)
# reg_cat.fit(X_train, y_train)

# # Calculate the predictions on the set set
# pred = reg_cat.predict(X_test)

# # Evaluate the performance using the RMSE
# rmse_cat = np.sqrt(mean_squared_error(y_test, pred))
# print('RMSE (CatBoost): {:.3f}'.format(rmse_cat))
# -

# ### Boosting contest: Light vs Extreme

# +
# import xgboost as xgb
# # Build and fit a XGBoost regressor
# reg_xgb = xgb.XGBRegressor(max_depth = 3, learning_rate = 0.1, n_estimators = 100, random_state=500)
# reg_xgb.fit(X_train, y_train)

# # Build and fit a LightGBM regressor
# reg_lgb = lgb.LGBMRegressor(max_depth = 3, learning_rate = 0.1, n_estimators = 100, seed=500)
# reg_lgb.fit(X_train, y_train)

# # Calculate the predictions and evaluate both regressors
# pred_xgb = reg_xgb.predict(X_test)
# rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
# pred_lgb = reg_lgb.predict(X_test)
# rmse_lgb = np.sqrt(mean_squared_error(y_test, pred_lgb))

# print('Extreme: {:.3f}, Light: {:.3f}'.format(rmse_xgb, rmse_lgb))
# -

# ## Stacking

# ### Predicting mushroom edibility

mr = pd.read_csv('mushrooms.csv')
y = pd.get_dummies(mr['class'])['p']
X = pd.get_dummies(mr.drop('class', axis = 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
mr.head()

# +
from sklearn.naive_bayes import GaussianNB
# Instantiate a Naive Bayes classifier
clf_nb = GaussianNB()

# Fit the model to the training set
clf_nb.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = clf_nb.predict(X_test)

# Evaluate the performance using the accuracy score
print("Accuracy: {:0.4f}".format(accuracy_score(y_test, pred)))
# -

# ### K-nearest neighbors for mushrooms

# +
# Instantiate a 5-nearest neighbors classifier with 'ball_tree' algorithm
clf_knn = KNeighborsClassifier(n_neighbors = 5, algorithm = 'ball_tree')

# Fit the model to the training set
clf_knn.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = clf_knn.predict(X_test)

# Evaluate the performance using the accuracy score
print("Accuracy: {:0.4f}".format(accuracy_score(y_test, pred)))
# -

# ### Applying stacking to predict app ratings

# +
# Build and fit a Decision Tree classifier
clf_dt = DecisionTreeClassifier(min_samples_leaf = 3, min_samples_split = 9, random_state=500)
clf_dt.fit(X_train, y_train)

# Build and fit a 5-nearest neighbors classifier using the 'Ball-Tree' algorithm
clf_knn = KNeighborsClassifier(n_neighbors = 5, algorithm = 'ball_tree')
clf_knn.fit(X_train, y_train)

# Evaluate the performance using the accuracy score
print('Decision Tree: {:0.4f}'.format(accuracy_score(y_test, clf_dt.predict(X_test))))
print('5-Nearest Neighbors: {:0.4f}'.format(accuracy_score(y_test, clf_knn.predict(X_test))))
# -

# ### A first attempt with mlxtend

# +
#conda install mlxtend --channel conda-forge
from mlxtend.classifier import StackingClassifier
# Instantiate the first-layer classifiers
clf_dt = DecisionTreeClassifier(min_samples_leaf=3, min_samples_split=9, random_state=500)
clf_knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')

# Instantiate the second-layer meta classifier
clf_meta = DecisionTreeClassifier(random_state=500)

# Build the Stacking classifier
clf_stack = StackingClassifier(classifiers=[clf_dt, clf_knn], meta_classifier=clf_meta, use_features_in_secondary=True)
clf_stack.fit(X_train, y_train)

# Evaluate the performance of the Stacking classifier
pred_stack = clf_stack.predict(X_test)
print("Accuracy: {:0.4f}".format(accuracy_score(y_test, pred_stack)))
# -

# ### Mushrooms: a matter of life or death

# +
# ERRO name 'StackingClassifier' is not defined, para corrigir
#conda update conda
#conda install scikit-learn=0.19

# Create the first-layer models
clf_knn = KNeighborsClassifier(n_neighbors = 5, algorithm = 'ball_tree')
clf_dt = DecisionTreeClassifier(min_samples_leaf = 5, min_samples_split = 15, random_state=500)
clf_nb = GaussianNB()

# Create the second-layer model (meta-model)
clf_lr = LogisticRegression(solver = 'liblinear')

# Create and fit the stacked model
clf_stack = StackingClassifier(classifiers = [clf_knn, clf_dt, clf_nb], meta_classifier = clf_lr)
clf_stack.fit(X_train, y_train)

# Evaluate the stacked model’s performance
print("Accuracy: {:0.4f}".format(accuracy_score(y_test, clf_stack.predict(X_test))))
# -

# ### Back to regression with stacking

X = app[['Reviews', 'Size', 'Installs', 'Paid', 'Price', 'Content Rating']]
y = app['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# +
from sklearn.linear_model import Ridge
from mlxtend.regressor import StackingRegressor
from sklearn.metrics import mean_absolute_error
# Instantiate the 1st-layer regressors
reg_dt = DecisionTreeRegressor(min_samples_leaf = 11, min_samples_split = 33, random_state=500)
reg_lr = LinearRegression(normalize = True)
reg_ridge = Ridge(random_state = 500)

# Instantiate the 2nd-layer regressor
reg_meta = LinearRegression()

# Build the Stacking regressor
reg_stack = StackingRegressor([reg_dt, reg_lr, reg_ridge], reg_meta)
reg_stack.fit(X_train, y_train)

# Evaluate the performance on the test set using the MAE metric
pred = reg_stack.predict(X_test)
print('MAE: {:.3f}'.format(mean_absolute_error(y_test, pred)))
# -






