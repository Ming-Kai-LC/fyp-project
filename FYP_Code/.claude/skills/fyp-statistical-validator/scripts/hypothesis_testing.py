#!/usr/bin/env python3
"""
Hypothesis Testing Suite for ML Model Comparison
Implements 3 tests required by TAR UMT thesis standards
"""

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from typing import Tuple, List, Dict


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_model1: np.ndarray,
    y_pred_model2: np.ndarray
) -> Tuple[float, float, str]:
    """
    McNemar's test for comparing two classifiers on same test set.
    
    Best for: Comparing CrossViT vs each baseline (ResNet-50, DenseNet-121, etc.)
    Tests null hypothesis: Two models have equal performance
    
    Contingency table:
                  Model 2 Correct    Model 2 Wrong
    Model 1 Correct      a                 b
    Model 1 Wrong        c                 d
    
    Test statistic: (b - c)² / (b + c)
    
    Args:
        y_true: array - true labels
        y_pred_model1: array - predictions from model 1 (e.g., CrossViT)
        y_pred_model2: array - predictions from model 2 (e.g., ResNet-50)
    
    Returns:
        (statistic, p_value, interpretation)
    
    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1, ...])
        >>> y_pred_crossvit = np.array([0, 1, 1, 1, 1, ...])
        >>> y_pred_resnet = np.array([0, 1, 0, 1, 1, ...])
        >>> stat, p, interp = mcnemar_test(y_true, y_pred_crossvit, y_pred_resnet)
        >>> print(f"McNemar: statistic={stat:.2f}, p={p:.4f}, {interp}")
    """
    # Check predictions are correct or wrong
    correct_1 = (y_pred_model1 == y_true)
    correct_2 = (y_pred_model2 == y_true)
    
    # Build contingency table
    # b = model1 correct, model2 wrong
    # c = model1 wrong, model2 correct
    b = np.sum(correct_1 & ~correct_2)
    c = np.sum(~correct_1 & correct_2)
    
    # McNemar's test requires b + c > 0
    if b + c == 0:
        return 0.0, 1.0, "Both models made identical predictions"
    
    # Create 2x2 table for statsmodels
    table = [[0, b], [c, 0]]
    
    # Apply McNemar's test with continuity correction
    result = mcnemar(table, exact=False, correction=True)
    
    # Interpretation
    if result.pvalue < 0.001:
        sig = "***"
        interpretation = f"Highly significant difference (p<0.001){sig}"
    elif result.pvalue < 0.01:
        sig = "**"
        interpretation = f"Very significant difference (p<0.01){sig}"
    elif result.pvalue < 0.05:
        sig = "*"
        interpretation = f"Significant difference (p<0.05){sig}"
    else:
        sig = ""
        interpretation = "No significant difference (p≥0.05)"
    
    return result.statistic, result.pvalue, interpretation


def paired_ttest(
    scores_model1: List[float],
    scores_model2: List[float],
    alpha: float = 0.05
) -> Tuple[float, float, bool, str]:
    """
    Paired t-test for comparing two models across multiple runs.
    
    Best for: Comparing 5 seeds of CrossViT vs 5 seeds of baseline
    Tests null hypothesis: Mean difference between paired scores is zero
    
    Requirements:
    - Same number of runs for both models
    - Runs paired by random seed (seed 1 vs seed 1, seed 2 vs seed 2, etc.)
    
    Args:
        scores_model1: list - scores from model 1 (e.g., [0.92, 0.93, 0.91, ...])
        scores_model2: list - scores from model 2 (e.g., [0.88, 0.89, 0.87, ...])
        alpha: float - significance level (default 0.05)
    
    Returns:
        (t_statistic, p_value, significant, interpretation)
    
    Example:
        >>> crossvit_seeds = [0.921, 0.925, 0.920, 0.928, 0.919]
        >>> resnet_seeds = [0.887, 0.892, 0.884, 0.895, 0.882]
        >>> t, p, sig, interp = paired_ttest(crossvit_seeds, resnet_seeds)
        >>> print(f"Paired t-test: t={t:.3f}, p={p:.4f}, {interp}")
    """
    scores_model1 = np.array(scores_model1)
    scores_model2 = np.array(scores_model2)
    
    if len(scores_model1) != len(scores_model2):
        raise ValueError("Both models must have same number of runs")
    
    if len(scores_model1) < 2:
        raise ValueError("Need at least 2 paired runs")
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(scores_model1, scores_model2)
    significant = p_value < alpha
    
    # Calculate mean difference
    mean_diff = np.mean(scores_model1) - np.mean(scores_model2)
    mean1 = np.mean(scores_model1)
    mean2 = np.mean(scores_model2)
    
    # Interpretation
    if significant:
        direction = "outperforms" if mean_diff > 0 else "underperforms"
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        else:
            sig = "*"
        interpretation = (f"Model 1 significantly {direction} Model 2 "
                        f"(mean difference: {mean_diff:+.1%}, p<{alpha}){sig}")
    else:
        interpretation = f"No significant difference (p≥{alpha})"
    
    return t_stat, p_value, significant, interpretation


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[List[bool], float, str]:
    """
    Bonferroni correction for multiple hypothesis tests.
    
    Best for: Comparing CrossViT against 5 baselines (5 comparisons)
    Adjusts significance threshold to control family-wise error rate
    
    Adjusted α = α / number_of_tests
    For 5 comparisons: 0.05 / 5 = 0.01
    
    Args:
        p_values: list - p-values from multiple tests
        alpha: float - desired family-wise error rate (default 0.05)
    
    Returns:
        (significant_flags, adjusted_alpha, interpretation)
    
    Example:
        >>> p_vals = [0.0001, 0.0023, 0.0089, 0.0234, 0.0456]
        >>> sig_flags, adj_alpha, interp = bonferroni_correction(p_vals)
        >>> print(f"Adjusted α: {adj_alpha:.4f}")
        >>> print(f"Significant tests: {sum(sig_flags)}/{len(sig_flags)}")
    """
    p_values = np.array(p_values)
    n_tests = len(p_values)
    
    if n_tests == 0:
        raise ValueError("Need at least one p-value")
    
    # Calculate adjusted alpha
    adjusted_alpha = alpha / n_tests
    
    # Check significance with adjusted threshold
    significant = p_values < adjusted_alpha
    
    # Interpretation
    n_significant = np.sum(significant)
    interpretation = (f"{n_significant}/{n_tests} tests remain significant "
                     f"after Bonferroni correction (adjusted α={adjusted_alpha:.4f})")
    
    return significant.tolist(), adjusted_alpha, interpretation


def delong_test(
    y_true: np.ndarray,
    y_scores_model1: np.ndarray,
    y_scores_model2: np.ndarray
) -> Tuple[float, float, str]:
    """
    DeLong's test for comparing two ROC curves (AUC comparison).
    
    Best for: Comparing AUC-ROC between CrossViT and baselines
    Tests null hypothesis: Two models have equal AUC
    
    Note: This is a simplified implementation. For production use,
    consider using scipy.stats or specialized libraries.
    
    Args:
        y_true: array - true binary labels
        y_scores_model1: array - predicted probabilities from model 1
        y_scores_model2: array - predicted probabilities from model 2
    
    Returns:
        (z_statistic, p_value, interpretation)
    
    Example:
        >>> from sklearn.metrics import roc_auc_score
        >>> auc1 = roc_auc_score(y_true, y_scores_model1)
        >>> auc2 = roc_auc_score(y_true, y_scores_model2)
        >>> z, p, interp = delong_test(y_true, y_scores_model1, y_scores_model2)
    """
    from sklearn.metrics import roc_auc_score
    
    # Calculate AUCs
    auc1 = roc_auc_score(y_true, y_scores_model1)
    auc2 = roc_auc_score(y_true, y_scores_model2)
    
    # Simplified variance estimation (for full implementation, use proper DeLong method)
    # This is an approximation for demonstration
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    n = len(y_true)
    
    # Approximate standard error
    se = np.sqrt((auc1 * (1 - auc1) + auc2 * (1 - auc2)) / n)
    
    # Z-test
    z_stat = (auc1 - auc2) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Interpretation
    if p_value < 0.05:
        direction = "higher" if auc1 > auc2 else "lower"
        interpretation = (f"Model 1 has significantly {direction} AUC than Model 2 "
                        f"(p<0.05, ΔAUC={abs(auc1-auc2):.3f})")
    else:
        interpretation = f"No significant difference in AUC (p≥0.05, ΔAUC={abs(auc1-auc2):.3f})"
    
    return z_stat, p_value, interpretation


def generate_comparison_report(
    model1_name: str,
    model2_name: str,
    test_results: Dict
) -> str:
    """
    Generate formatted comparison report for thesis.
    
    Args:
        model1_name: str - name of model 1 (e.g., "CrossViT")
        model2_name: str - name of model 2 (e.g., "ResNet-50")
        test_results: dict - results from hypothesis tests
    
    Returns:
        Formatted report string
    
    Example:
        >>> results = {
        ...     'mcnemar': (15.23, 0.0001, "Significant"),
        ...     'paired_t': (4.23, 0.013, "Significant")
        ... }
        >>> report = generate_comparison_report("CrossViT", "ResNet-50", results)
    """
    report = f"\n{'='*70}\n"
    report += f"HYPOTHESIS TESTING: {model1_name} vs {model2_name}\n"
    report += f"{'='*70}\n\n"
    
    if 'mcnemar' in test_results:
        stat, p, interp = test_results['mcnemar']
        report += f"McNemar's Test:\n"
        report += f"  Statistic: {stat:.2f}\n"
        report += f"  p-value: {p:.4f}\n"
        report += f"  Result: {interp}\n\n"
    
    if 'paired_t' in test_results:
        t, p, sig, interp = test_results['paired_t']
        report += f"Paired t-test:\n"
        report += f"  t-statistic: {t:.3f}\n"
        report += f"  p-value: {p:.4f}\n"
        report += f"  Result: {interp}\n\n"
    
    if 'delong' in test_results:
        z, p, interp = test_results['delong']
        report += f"DeLong's Test (AUC comparison):\n"
        report += f"  z-statistic: {z:.3f}\n"
        report += f"  p-value: {p:.4f}\n"
        report += f"  Result: {interp}\n\n"
    
    report += f"{'='*70}\n"
    
    return report


if __name__ == "__main__":
    # Example usage
    print("=== Hypothesis Testing Suite Demo ===\n")
    
    # Simulate data
    np.random.seed(42)
    n = 2117
    y_true = np.random.randint(0, 2, n)
    
    # Model 1: CrossViT (92.3% accuracy)
    y_pred_crossvit = y_true.copy()
    error_idx1 = np.random.choice(n, size=int(n * 0.077), replace=False)
    y_pred_crossvit[error_idx1] = 1 - y_pred_crossvit[error_idx1]
    
    # Model 2: ResNet-50 (88.7% accuracy)
    y_pred_resnet = y_true.copy()
    error_idx2 = np.random.choice(n, size=int(n * 0.113), replace=False)
    y_pred_resnet[error_idx2] = 1 - y_pred_resnet[error_idx2]
    
    # Test 1: McNemar's Test
    print("1. McNemar's Test (Single Run):")
    stat, p, interp = mcnemar_test(y_true, y_pred_crossvit, y_pred_resnet)
    print(f"   Statistic: {stat:.2f}")
    print(f"   p-value: {p:.4f}")
    print(f"   {interp}\n")
    
    # Test 2: Paired t-test
    print("2. Paired t-test (5 Seeds Each):")
    crossvit_seeds = [0.921, 0.925, 0.920, 0.928, 0.919]
    resnet_seeds = [0.887, 0.892, 0.884, 0.895, 0.882]
    t, p, sig, interp = paired_ttest(crossvit_seeds, resnet_seeds)
    print(f"   t-statistic: {t:.3f}")
    print(f"   p-value: {p:.4f}")
    print(f"   {interp}\n")
    
    # Test 3: Bonferroni Correction
    print("3. Bonferroni Correction (5 Comparisons):")
    p_values = [0.0001, 0.0023, 0.0089, 0.0234, 0.0456]
    sig_flags, adj_alpha, interp = bonferroni_correction(p_values)
    print(f"   Original α: 0.05")
    print(f"   Adjusted α: {adj_alpha:.4f}")
    print(f"   {interp}")
    print(f"   Significant tests: {[i+1 for i, s in enumerate(sig_flags) if s]}")
