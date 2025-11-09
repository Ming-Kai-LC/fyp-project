"""
Model Training and Evaluation Module
Reusable functions for model training, evaluation, and validation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy import stats
import joblib


def train_model(model, X_train, y_train, X_val=None, y_val=None):
    """
    Train a machine learning model.

    Parameters:
    -----------
    model : sklearn estimator
        Model to train
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training target
    X_val : pd.DataFrame or np.ndarray, optional
        Validation features
    y_val : pd.Series or np.ndarray, optional
        Validation target

    Returns:
    --------
    model
        Trained model
    """
    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        val_score = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {val_score:.4f}")

    return model


def evaluate_classification(y_true, y_pred, y_pred_proba=None):
    """
    Evaluate classification model performance.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities

    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }

    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        except:
            pass

    print("Classification Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return metrics


def evaluate_regression(y_true, y_pred):
    """
    Evaluate regression model performance.

    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

    print("Regression Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics


def calculate_confidence_interval(scores, confidence=0.95):
    """
    Calculate confidence interval for model scores.

    Parameters:
    -----------
    scores : array-like
        Cross-validation scores
    confidence : float
        Confidence level (default 0.95)

    Returns:
    --------
    tuple
        (mean, lower_bound, upper_bound)
    """
    n = len(scores)
    mean = np.mean(scores)
    std_error = stats.sem(scores)
    margin = std_error * stats.t.ppf((1 + confidence) / 2, n - 1)

    lower = mean - margin
    upper = mean + margin

    print(f"Mean Score: {mean:.4f}")
    print(f"95% CI: [{lower:.4f}, {upper:.4f}]")

    return mean, lower, upper


def compare_models(model1, model2, X, y, cv=10):
    """
    Compare two models using statistical testing.

    Parameters:
    -----------
    model1 : sklearn estimator
        First model
    model2 : sklearn estimator
        Second model
    X : pd.DataFrame or np.ndarray
        Features
    y : pd.Series or np.ndarray
        Target
    cv : int
        Number of cross-validation folds

    Returns:
    --------
    dict
        Comparison results including t-statistic, p-value, and effect size
    """
    scores1 = cross_val_score(model1, X, y, cv=cv)
    scores2 = cross_val_score(model2, X, y, cv=cv)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores1, scores2)

    # Cohen's d effect size
    mean_diff = np.mean(scores1) - np.mean(scores2)
    pooled_std = np.sqrt((np.std(scores1, ddof=1)**2 + np.std(scores2, ddof=1)**2) / 2)
    cohens_d = mean_diff / pooled_std

    print(f"Model 1 mean score: {np.mean(scores1):.4f}")
    print(f"Model 2 mean score: {np.mean(scores2):.4f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Cohen's d: {cohens_d:.4f}")

    if p_value < 0.05:
        print("✓ Statistically significant difference (p < 0.05)")
    else:
        print("✗ No significant difference")

    return {
        'model1_scores': scores1,
        'model2_scores': scores2,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d
    }


def hyperparameter_tuning(model, param_grid, X_train, y_train, cv=5, scoring='accuracy'):
    """
    Perform grid search for hyperparameter tuning.

    Parameters:
    -----------
    model : sklearn estimator
        Model to tune
    param_grid : dict
        Parameter grid for grid search
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training target
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric

    Returns:
    --------
    sklearn estimator
        Best model from grid search
    """
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def save_model(model, filepath):
    """
    Save trained model to disk.

    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    filepath : str
        Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load trained model from disk.

    Parameters:
    -----------
    filepath : str
        Path to the saved model

    Returns:
    --------
    sklearn estimator
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
