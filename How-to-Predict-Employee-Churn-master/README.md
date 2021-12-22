# How to Predict Employee Churn

## Project Motivation

The ability to predict employee churn is powerful. Because you would have the opportunity to retain good employees, and also prepare for recruitment. This project used a machine learning model to predict employee churn and identified features most important for this task. I followed the DRISP-DM process and addressed three questions:

- What is the employee churn rate?
- How important is each feature in predicting the employee churn?
- How is each feature used in the prediction?

## Document Description
The raw data of this project is stored in [turnover.csv](https://github.com/iDataist/How-to-Predict-Employee-Churn/blob/master/turnover.csv), which was collected by the HR team of a company. The data analysis, modeling and visualization is documented in [this](https://github.com/iDataist/How-to-Predict-Employee-Churn/blob/master/Predicting_Employee_Churn.ipynb) notebook. The findings are communicated in my personal [blog](https://idataist.com/2019/09/28/how-to-predict-employee-churn-with-machine-learning/?preview_id=513&preview_nonce=6174e240e5&preview=true).

## Libraries Used
- pandas
- numpy
- sklearn
- matplotlib
- seaborn
- IPython
- pydotplus

## Project Results
- Based on the data collected by the HR team, the employee churn rate of the company was 23.8%.
- The most important features are satisfaction, time spend with the company, evaluation, number of projects and the average monthly hours, ranked from the most to the least important.
- Satisfaction was the first feature used to separate the samples. Then time spend with the company and evaluation are the main features that separate samples in the second and third rounds. 
