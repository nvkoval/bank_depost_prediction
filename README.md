# Predict the Success of Bank Marketing Campaigns
This repository contains a Machine Learning pipeline for predict the success of bank marketing using various classification models, including Logistic Regression, Random Forest, and Boosting methods. The pipeline includes data preprocessing, feature engineering, model training, evaluation, and visualization.

## Project Overview
The goal of this project is to predict whether a bank customer will subscribe to a term deposit based on demographic and financial attributes. The key objectives are:
- Build a baseline model with Logistic Regression.
- Compare performance with Random Forest and Boosting models.
- Optimize the feature engineering and preprocessing pipeline.
- Use AUC Score as the primary evaluation metric.
- Interpret the results using a confusion matrix and confidence analysis.

## Dataset
The data from the UCI Machine Learning Repository is related with direct marketing campaigns (phone calls) of a Portuguese banking institution.
The dataset consists of customer data with features such as:
- Demographic information (age, job, marital status, education, etc.)
- Previous interactions (campaign contacts, previous outcomes, etc.)
- Economic indicators (employment variation rate, consumer price index, etc.)
- Target variable: y (binary: 'yes' = 1, 'no' = 0)

## Project Overview
The project includes full-cycle machine learning development:
- Feature engineering
- Preprocessing pipelines
- Handling class imbalance
- Model training and evaluation
- Hyperparameter tuning

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

## Results
| Model Name             | AUROC on Train | AUROC on Validation |
| ---------------------- | -------------- | ------------------- |
| LogisticRegression     |0.797910        |0.791954             |
| KNeighborsClassifier   |0.869958        |0.746194             |
| DecisionTreeClassifier |0.791737        |0.776121             |
| RandomForestClassifier |0.857783        |0.80122              |
| XGBClassifier          |0.837919        |0.80117              |
| LGBMClassifier         |0.820159        |0.801564             |

## Future Work
- Try advanced feature selection techniques.
- Optimize the decision threshold based on the precision-recall trade-off.
- Experiment with deep learning approaches.
- Deploy the model as an API.

## Repository Structure
```
├── data/                              # Folder for dataset and model files
├── bank_deposit_EDA.ipynb             # Exploratory Data Analysis (EDA) notebook
├── bank_deposit.ipynb                 # Main modeling and training notebook
├── bank_deposit_resampling.ipynb      # Notebook for resampling experiments
├── bank_deposit_Error_Analysis.ipynb  # Error analysis notebook
├── process_bank_deposit.py            # Python script for data processing and model pipeline
├── utils.py                           # Utility functions for the project
├── requirements.txt                   # List of required Python packages
└── README.md                          # Project documentation
```
