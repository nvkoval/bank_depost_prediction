# Customer Churn Prediction
This repository contains a Machine Learning pipeline for bank customer churn prediction using various classification models, including Logistic Regression, Random Forest, and Boosting methods. The pipeline includes data preprocessing, feature engineering, model training, evaluation, and visualization.


## Project Overview
The goal of this project is to predict whether a bank customer will subscribe to a term deposit based on demographic and financial attributes. The key objectives are:
- Build a baseline model with Logistic Regression.
- Compare performance with Random Forest and Boosting models.
- Optimize the feature engineering and preprocessing pipeline.
- Use AUC Score as the primary evaluation metric.
- Interpret the results using a confusion matrix and confidence analysis.

## Dataset
The dataset consists of customer data with features such as:
- Demographic information (age, job, marital status, education, etc.)
- Previous interactions (campaign contacts, previous outcomes, etc.)
- Economic indicators (employment variation rate, consumer price index, etc.)
- Target variable: y (binary: 'yes' = 1, 'no' = 0)

## Feature Engineering
The following features were engineered to improve predictive power:
- Categorical Features: Ordinal encoding for education, one-hot encoding for others.
- New Features: Age groups, number of contacts, recent contact status, and economic ratios.

## Preprocessing Pipeline
The preprocessing steps include:
- Handling categorical and numerical features differently for tree-based and linear models.
- Applying MinMax Scaling for Logistic Regression.
- Using Ordinal and One-Hot Encoding for categorical variables.
- Transforming economic indicators into new ratio-based features.

## Model Training and Evaluation
The models are trained using cross-validation. Evaluation includes:
- AUC Score: Main metric for model performance.
- Confusion Matrix: To analyze misclassifications.

## Future Work
- Try advanced feature selection techniques.
- Optimize the decision threshold based on the precision-recall trade-off.
- Experiment with deep learning approaches.
- Deploy the model as an API.
