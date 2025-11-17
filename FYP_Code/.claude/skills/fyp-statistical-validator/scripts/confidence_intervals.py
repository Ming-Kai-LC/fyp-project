#!/usr/bin/env python3
"""
Confidence Interval Calculators for Machine Learning Experiments
Implements 3 methods required by TAR UMT thesis standards
"""

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
from typing import Tuple, List


def normal_approximation_ci(
    accuracy: float, 
    n_samples: int, 
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval using normal approximation.
    
    Best for: Single train/test split, quick estimates, n > 30
    Formula: ACC ± z × √[(ACC × (1-ACC)) / n]
    
    Args:
        accuracy: float (0-1) - model accuracy
        n_samples: int - test set size
        confidence: float - confidence level (default 0.95)
    
    Returns:
        (mean, lower_bound, upper_bound)
    
    Example:
        >>> mean, lower, upper = normal_approximation_ci(0.923, 2117)
        >>> print(f"{mean:.1%} (95% CI: {lower:.1%} - {upper:.1%})")
        92.3% (95% CI: 91.1% - 93.5%)
    """
    # Calculate z-score for confidence level
    alpha = 1 - confidence
    z_score = stats.norm.ppf(1 - alpha / 2)
    
    # Standard error
    se = np.sqrt((accuracy * (1 - accuracy)) / n_samples)
    
    # Confidence interval
    margin = z_score * se
    lower_bound = accuracy - margin
    upper_bound = accuracy + margin
    
    # Clip to [0, 1] range
    lower_bound = max(0.0, lower_bound)
    upper_bound = min(1.0, upper_bound)
    
    return accuracy, lower_bound, upper_bound


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval using bootstrap resampling.
    
    Best for: Robust CI when distribution unknown, non-normal distributions
    More computationally intensive but more robust than normal approximation
    
    Args:
        y_true: array - true labels
        y_pred: array - predicted labels
        n_iterations: int - bootstrap rounds (default 1000)
        confidence: float - confidence level
        random_state: int - random seed for reproducibility
    
    Returns:
        (mean, lower_bound, upper_bound)
    
    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1, ...])
        >>> y_pred = np.array([0, 1, 1, 1, 1, ...])
        >>> mean, lower, upper = bootstrap_ci(y_true, y_pred)
        >>> print(f"Bootstrap CI: {mean:.1%} (95% CI: {lower:.1%} - {upper:.1%})")
    """
    np.random.seed(random_state)
    
    n = len(y_true)
    scores = []
    
    for _ in range(n_iterations):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        score = accuracy_score(y_true[indices], y_pred[indices])
        scores.append(score)
    
    scores = np.array(scores)
    
    # Percentile method
    alpha = (1 - confidence) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)
    mean = np.mean(scores)
    
    return mean, lower, upper


def multi_seed_ci(
    accuracies: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float, float]:
    """
    Calculate confidence interval from multiple independent training runs.
    
    Best for: Comparing models with multiple random seeds
    TAR UMT requirement: Minimum 5 seeds recommended
    Uses t-distribution for small sample sizes
    
    Args:
        accuracies: list/array - accuracy from each seed
        confidence: float - confidence level
    
    Returns:
        (mean, lower_bound, upper_bound, std_dev)
    
    Example:
        >>> seeds_results = [0.921, 0.925, 0.920, 0.928, 0.919]
        >>> mean, lower, upper, std = multi_seed_ci(seeds_results)
        >>> print(f"{mean:.1%} ± {std:.1%} (95% CI: {lower:.1%} - {upper:.1%})")
        92.3% ± 1.2% (95% CI: 90.8% - 93.8%)
    """
    accuracies = np.array(accuracies)
    n = len(accuracies)
    
    if n < 2:
        raise ValueError("Need at least 2 runs for confidence interval")
    
    mean = np.mean(accuracies)
    std = np.std(accuracies, ddof=1)  # Sample standard deviation
    se = std / np.sqrt(n)
    
    # Use t-distribution for small samples
    df = n - 1
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha / 2, df=df)
    
    margin = t_value * se
    lower_bound = mean - margin
    upper_bound = mean + margin
    
    # Clip to [0, 1] range
    lower_bound = max(0.0, lower_bound)
    upper_bound = min(1.0, upper_bound)
    
    return mean, lower_bound, upper_bound, std


def format_ci_result(
    mean: float,
    lower: float,
    upper: float,
    std: float = None,
    metric_name: str = "Accuracy"
) -> str:
    """
    Format confidence interval results for thesis presentation.
    
    Args:
        mean: float - mean metric value
        lower: float - lower bound
        upper: float - upper bound
        std: float - standard deviation (optional)
        metric_name: str - name of the metric
    
    Returns:
        Formatted string ready for thesis
    
    Example:
        >>> result = format_ci_result(0.923, 0.911, 0.935)
        >>> print(result)
        Accuracy: 92.3% (95% CI: 91.1% - 93.5%)
    """
    if std is not None:
        return (f"{metric_name}: {mean:.1%} ± {std:.1%} "
                f"(95% CI: {lower:.1%} - {upper:.1%})")
    else:
        return f"{metric_name}: {mean:.1%} (95% CI: {lower:.1%} - {upper:.1%})"


if __name__ == "__main__":
    # Example usage
    print("=== Confidence Interval Calculator Demo ===\n")
    
    # Method 1: Normal Approximation (single test)
    print("1. Normal Approximation (Single Split):")
    mean, lower, upper = normal_approximation_ci(0.923, 2117)
    print(format_ci_result(mean, lower, upper))
    print()
    
    # Method 2: Bootstrap (with data)
    print("2. Bootstrap Resampling:")
    # Simulated predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 2117)
    y_pred = y_true.copy()
    # Add some errors (7.7% error rate = 92.3% accuracy)
    error_indices = np.random.choice(2117, size=int(2117 * 0.077), replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]
    
    mean, lower, upper = bootstrap_ci(y_true, y_pred, n_iterations=1000)
    print(format_ci_result(mean, lower, upper))
    print()
    
    # Method 3: Multi-Seed Aggregation
    print("3. Multi-Seed Aggregation (5 runs):")
    seeds_results = [0.921, 0.925, 0.920, 0.928, 0.919]
    mean, lower, upper, std = multi_seed_ci(seeds_results)
    print(format_ci_result(mean, lower, upper, std))
    print(f"Individual seeds: {[f'{x:.1%}' for x in seeds_results]}")
