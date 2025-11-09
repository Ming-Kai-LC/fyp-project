"""
Feature Engineering Module
Reusable functions for creating and transforming features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def create_interaction_features(df, feature_pairs):
    """
    Create interaction features from pairs of columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_pairs : list of tuples
        List of (feature1, feature2) pairs to create interactions

    Returns:
    --------
    pd.DataFrame
        Dataframe with interaction features added
    """
    df_copy = df.copy()

    for feat1, feat2 in feature_pairs:
        df_copy[f'{feat1}_x_{feat2}'] = df_copy[feat1] * df_copy[feat2]

    return df_copy


def create_ratio_features(df, numerator_denominator_pairs):
    """
    Create ratio features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numerator_denominator_pairs : list of tuples
        List of (numerator, denominator) column pairs

    Returns:
    --------
    pd.DataFrame
        Dataframe with ratio features added
    """
    df_copy = df.copy()

    for num, denom in numerator_denominator_pairs:
        df_copy[f'{num}_div_{denom}'] = df_copy[num] / (df_copy[denom] + 1e-10)

    return df_copy


def create_polynomial_features(X, degree=2):
    """
    Create polynomial features.

    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Input features
    degree : int
        Degree of polynomial features

    Returns:
    --------
    tuple
        (X_poly, poly_transformer)
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    return X_poly, poly


def bin_feature(df, column, bins, labels=None):
    """
    Bin a continuous feature into categories.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to bin
    bins : int or list
        Number of bins or list of bin edges
    labels : list, optional
        Labels for bins

    Returns:
    --------
    pd.DataFrame
        Dataframe with binned feature added
    """
    df_copy = df.copy()
    df_copy[f'{column}_binned'] = pd.cut(df_copy[column], bins=bins, labels=labels)

    return df_copy


def extract_datetime_features(df, datetime_column):
    """
    Extract features from datetime column.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    datetime_column : str
        Name of datetime column

    Returns:
    --------
    pd.DataFrame
        Dataframe with datetime features extracted
    """
    df_copy = df.copy()
    df_copy[datetime_column] = pd.to_datetime(df_copy[datetime_column])

    df_copy[f'{datetime_column}_year'] = df_copy[datetime_column].dt.year
    df_copy[f'{datetime_column}_month'] = df_copy[datetime_column].dt.month
    df_copy[f'{datetime_column}_day'] = df_copy[datetime_column].dt.day
    df_copy[f'{datetime_column}_dayofweek'] = df_copy[datetime_column].dt.dayofweek
    df_copy[f'{datetime_column}_is_weekend'] = df_copy[datetime_column].dt.dayofweek.isin([5, 6]).astype(int)

    return df_copy
