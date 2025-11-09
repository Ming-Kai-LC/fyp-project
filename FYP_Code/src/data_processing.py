"""
Data Processing Module
Reusable functions for data cleaning and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer

def load_data(file_path, **kwargs):
    """
    Load data from various file formats.

    Parameters:
    -----------
    file_path : str
        Path to the data file
    **kwargs : dict
        Additional arguments for pandas read functions

    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, **kwargs)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path, **kwargs)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def handle_missing_values(df, strategy='median', columns=None):
    """
    Handle missing values using specified strategy.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        Imputation strategy ('median', 'mean', 'mode', 'knn')
    columns : list, optional
        Specific columns to impute. If None, impute all numeric columns

    Returns:
    --------
    pd.DataFrame
        Dataframe with imputed values
    """
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns

    if strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:
        imputer = SimpleImputer(strategy=strategy)

    df_copy[columns] = imputer.fit_transform(df_copy[columns])

    return df_copy


def remove_outliers(df, columns, method='isolation_forest', contamination=0.05):
    """
    Remove outliers using specified method.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to check for outliers
    method : str
        Method to use ('isolation_forest', 'iqr', 'zscore')
    contamination : float
        Proportion of outliers in dataset (for isolation_forest)

    Returns:
    --------
    pd.DataFrame
        Dataframe with outliers removed
    """
    from sklearn.ensemble import IsolationForest

    df_copy = df.copy()

    if method == 'isolation_forest':
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(df_copy[columns])
        df_copy = df_copy[outliers == 1]

    elif method == 'iqr':
        for col in columns:
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_copy = df_copy[(df_copy[col] >= lower) & (df_copy[col] <= upper)]

    elif method == 'zscore':
        from scipy import stats
        for col in columns:
            z_scores = np.abs(stats.zscore(df_copy[col]))
            df_copy = df_copy[z_scores < 3]

    return df_copy


def scale_features(X_train, X_test, method='standard'):
    """
    Scale features using specified method.

    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    X_test : pd.DataFrame or np.ndarray
        Test features
    method : str
        Scaling method ('standard', 'minmax', 'robust')

    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
