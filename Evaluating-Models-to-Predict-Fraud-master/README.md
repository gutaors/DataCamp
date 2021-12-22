# Evaluating-Models-to-Predict-Fraud
This project trained several models and evaluated how effectively they predict instances of fraud. The results are presented in [Model_Evaluation.ipynb](https://github.com/iDataist/Evaluate-Models-to-Predict-Fraud/blob/master/Model_Evaluation.ipynb). 

The raw data is stored in [fraud_data.csv](https://github.com/iDataist/Evaluate-Models-to-Predict-Fraud/blob/master/fraud_data.csv). Each row corresponds to a credit card transaction. Features include confidential variables `V1` through `V28` as well as `Amount` which is the amount of the transaction. The target is stored in the `class` column, where a value of 1 corresponds to an instance of fraud and 0 corresponds to an instance of not fraud.
