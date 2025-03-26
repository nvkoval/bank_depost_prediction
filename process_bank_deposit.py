from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (cross_validate, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def split_data(
        raw_df: pd.DataFrame,
        target_col: str
) -> Dict[str, pd.DataFrame]:
    """
    Split the raw dataframe into training and test sets.

    Args:
        raw_df (pd.DataFrame): The raw dataframe.
        target_col (str): Target column.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing the train
            and test dataframes.
    """
    train_df, test_df = train_test_split(
        raw_df.drop(columns=['duration']),
        test_size=0.2,
        stratify=raw_df[target_col],
        random_state=42
    )
    return {'train': train_df, 'test': test_df}


def create_inputs_and_targets(
        df_dict: Dict[str, pd.DataFrame],
        input_cols: list[str],
        target_col: str
) -> Dict[str, Any]:
    """
    Create inputs and targets for training and test sets.

    Args:
        df_dict (Dict[str, pd.DataFrame]): Dictionary containing the train
            and test dataframes.
        input_cols (list): List of input columns.
        target_col (str): Target column.

    Returns:
        Dict[str, Any]: Dictionary containing inputs and targets
            for train and test sets.
    """
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = (
            df_dict[split][target_col].map({'yes': 1, 'no': 0}).copy()
        )
    return data


def get_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numerical and categorical columns in the dataframe.
    """
    numeric_cols = df.select_dtypes(include='number').columns.to_list()
    categorical_cols = df.select_dtypes(include='object').columns.to_list()
    return numeric_cols, categorical_cols


def feature_engineering(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering transformations.
    """
    inputs = inputs.copy()
    inputs['age_cat'] = inputs['age'] // 10
    inputs['is_only_one_contact'] = (inputs['campaign'] == 1).astype(int)
    inputs['more_than_six_contacts'] = (inputs['campaign'] > 6).astype(int)
    inputs['previous_contact'] = (inputs['previous'] > 0).astype(int)
    inputs['recent_contact'] = (inputs['pdays'] < 7).astype(int)
    inputs['pdays_3'] = (inputs['pdays'] == 3).astype(int)
    inputs['pdays_6'] = (inputs['pdays'] == 6).astype(int)
    inputs['is_hight_education'] = (inputs['education']
                                    .isin(['university.degree',
                                           'professional.course'])
                                    .astype(int))
    inputs['is_basic_education'] = (inputs['education']
                                    .isin(['basic.4y', 'basic.9y',
                                           'basic.6y'])
                                    .astype(int))
    inputs['nr.employed_to_emp.var.rate'] = (inputs['nr.employed']
                                             / inputs['emp.var.rate'])
    inputs['cons.price.idx_to_cons.conf.idx'] = (inputs['cons.price.idx']
                                                 / inputs['cons.conf.idx'])
    inputs['euribor3m_to_emp.var.rate'] = (inputs['euribor3m']
                                           / inputs['emp.var.rate'])
    inputs['cons.price.idx_to_emp.var.rate'] = (inputs['cons.price.idx']
                                                / inputs['emp.var.rate'])
    inputs['cons.price.idx_to_euribor3m'] = (inputs['cons.price.idx']
                                             / inputs['euribor3m'])
    inputs['cons.conf.idx_to_euribor3m'] = (inputs['cons.conf.idx']
                                            / inputs['euribor3m'])
    return inputs


def get_train_test_data(
        raw_df: pd.DataFrame,
        target_col: str
) -> Dict[str, Any]:
    """
    Create inputs and targets for training and test sets.

    Args:
        df_dict (Dict[str, pd.DataFrame]): Dictionary containing the train
            and test dataframes.
        input_cols (list): List of input columns.
        target_col (str): Target column.

    Returns:
        Dict[str, Any]: Dictionary containing inputs and targets
            for train and test sets.
    """
    split_data_dict = split_data(raw_df, target_col)

    input_cols = list(raw_df.drop(columns=['duration', 'y']).columns)
    for split in split_data_dict:
        split_data_dict[split] = feature_engineering(split_data_dict[split])
    data_dict = create_inputs_and_targets(
        split_data_dict, input_cols, target_col
    )
    return data_dict


def get_preprocessor(inputs: pd.DataFrame,
                     tree_base_model: bool = True) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for numerical and categorical features.
    """
    numeric_cols, categorical_cols = get_feature_types(inputs)

    scaler = MinMaxScaler()

    one_hot_encoder = OneHotEncoder(
        sparse_output=False, handle_unknown='ignore'
    )

    ordinal_mapping_education = [[
        'unknown', 'illiterate', 'basic.4y', 'basic.6y',
        'basic.9y', 'high.school', 'professional.course',
        'university.degree']]

    ordinal_enc = OrdinalEncoder(categories=ordinal_mapping_education)

    if tree_base_model:
        one_hot_enc_cols = [
            'job', 'marital', 'default', 'housing', 'loan',
            'contact', 'month', 'day_of_week', 'poutcome', 'previous']
        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot_enc', one_hot_encoder, one_hot_enc_cols),
                ('ord_enc', ordinal_enc, ['education'])
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform='pandas')
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', scaler, numeric_cols),
                ('onehot_enc', one_hot_encoder, categorical_cols)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform='pandas')
    preprocessor.fit(inputs)
    return preprocessor


def preprocess_data(
        raw_df: pd.DataFrame,
        target_col: str,
        tree_base_model: bool = True
        ) -> Dict[str, Any]:
    """
    Preprocess the raw dataframe.

    Args:
        raw_df (pd.DataFrame): The raw dataframe.
        target_col (str): Target column.
        scaler_numeric (bool): Whether to scale numeric features.
            Default is True.
        impute_strategy (str): Strategy for imputing missing values.
            Default is 'median'.

    Returns:
        Dict[str, Any]: Dictionary containing processed inputs and targets
            for train and validation sets.
    """
    split_data_dict = split_data(raw_df, target_col)

    input_cols = list(raw_df.drop(columns=['duration', 'y']).columns)
    data_dict = create_inputs_and_targets(
        split_data_dict, input_cols, target_col
    )
    train_inputs = data_dict['train_inputs']
    train_targets = data_dict['train_targets']

    train_inputs = feature_engineering(train_inputs)
    preprocessor = get_preprocessor(train_inputs, tree_base_model)
    train_inputs_transform = preprocessor.transform(train_inputs)
    return train_inputs_transform, train_targets


def preprocess_new_data(
        new_data: pd.DataFrame,
        preprocessor: ColumnTransformer) -> pd.DataFrame:
    """
    Preprocess new incoming data using the trained preprocessor.
    """
    return preprocessor.transform(new_data)


def get_auc(model, inputs: pd.DataFrame, targets):
    """
    Compute AUROC score for rhe given model
    """
    auc = cross_validate(model,
                         X=inputs,
                         y=targets,
                         scoring='roc_auc',
                         cv=3,
                         return_train_score=True)
    train_score = np.mean(auc['train_score'])
    val_score = np.mean(auc['test_score'])
    print(f"AUROC score on train set:: {train_score:.3f}")
    print(f"AUROC score on validation set:: {val_score:.3f}")
    return train_score, val_score


def get_eval_results(preprocessor, clf, inputs, targets):
    """
    Evaluate the model using AUROC
    """
    # dictionary for saving results
    results_dict = {'model_name': str(clf).split('(')[0],
                    'params': str(clf).split('(')[1].rsplit(')')[0]}

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    train_score, val_score = get_auc(model_pipeline, inputs, targets)
    results_dict['AUROC on train'] = train_score
    results_dict['AUROC on validation'] = val_score
    return results_dict


def get_confusion_matrix(model, inputs, targets):
    """
    Generate and plot the Confusion matrix.
    """
    preds = cross_val_predict(model, inputs, targets, cv=3)
    cm = confusion_matrix(targets, preds, normalize='true')
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, cmap='YlOrBr',
                xticklabels=['no', 'yes'], yticklabels=['no', 'yes'])
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('Confusion Matrix')
    plt.show()
