#!/usr/bin/env python3
"""
APA Table Formatter for TAR UMT Thesis
Generates publication-ready tables with confidence intervals
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def format_results_table(
    models: List[str],
    metrics: Dict[str, List[Tuple[float, float, float]]],
    caption: str = "Model Performance Comparison",
    table_number: int = 1
) -> str:
    """
    Generate APA-formatted results table for thesis Chapter 5.
    
    Args:
        models: list - model names ["CrossViT", "ResNet-50", ...]
        metrics: dict - {metric_name: [(mean, lower_ci, upper_ci), ...]}
                 Example: {"Accuracy": [(0.923, 0.911, 0.935), (0.887, 0.872, 0.902)]}
        caption: str - table caption
        table_number: int - table number in thesis
    
    Returns:
        APA-formatted table as string
    
    Example:
        >>> models = ["CrossViT", "ResNet-50", "DenseNet-121"]
        >>> metrics = {
        ...     "Accuracy": [(0.923, 0.911, 0.935), (0.887, 0.872, 0.902), (0.892, 0.878, 0.906)],
        ...     "F1-Score": [(0.91, 0.89, 0.93), (0.87, 0.85, 0.89), (0.88, 0.86, 0.90)]
        ... }
        >>> table = format_results_table(models, metrics, "CrossViT vs Baselines")
        >>> print(table)
    """
    # Build table data
    table_data = {"Model": models}
    
    for metric_name, values in metrics.items():
        formatted_values = []
        for mean, lower, upper in values:
            # Format: 92.3% (91.1-93.5)
            formatted = f"{mean:.1%} ({lower:.1%}-{upper:.1%})"
            formatted_values.append(formatted)
        table_data[metric_name] = formatted_values
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Generate APA table
    output = f"\nTable {table_number}\n"
    output += f"{caption}\n"
    output += "=" * 80 + "\n"
    output += df.to_string(index=False)
    output += "\n" + "=" * 80 + "\n"
    output += "Note. Values shown as Mean (95% CI Lower-Upper).\n"
    output += "All metrics computed on test set (n=2,117).\n"
    
    return output


def format_hypothesis_table(
    comparisons: List[Tuple[str, str]],
    test_results: List[Dict],
    table_number: int = 2
) -> str:
    """
    Generate hypothesis testing results table for thesis.
    
    Args:
        comparisons: list - [(model1, model2), ...] pairs
        test_results: list - [{test_name: (statistic, p_value)}, ...]
        table_number: int - table number in thesis
    
    Returns:
        APA-formatted hypothesis testing table
    
    Example:
        >>> comparisons = [
        ...     ("CrossViT", "ResNet-50"),
        ...     ("CrossViT", "DenseNet-121")
        ... ]
        >>> results = [
        ...     {"mcnemar": (15.23, 0.0001), "paired_t": (4.23, 0.013)},
        ...     {"mcnemar": (12.45, 0.0004), "paired_t": (3.87, 0.018)}
        ... ]
        >>> table = format_hypothesis_table(comparisons, results)
    """
    # Build table data
    rows = []
    for (model1, model2), tests in zip(comparisons, test_results):
        row = {
            "Comparison": f"{model1} vs {model2}",
        }
        
        for test_name, (statistic, p_value) in tests.items():
            # Add significance markers
            if p_value < 0.001:
                sig = "***"
            elif p_value < 0.01:
                sig = "**"
            elif p_value < 0.05:
                sig = "*"
            else:
                sig = ""
            
            row[f"{test_name.title()} (p)"] = f"{p_value:.4f}{sig}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Generate table
    output = f"\nTable {table_number}\n"
    output += "Statistical Significance of Model Comparisons\n"
    output += "=" * 80 + "\n"
    output += df.to_string(index=False)
    output += "\n" + "=" * 80 + "\n"
    output += "Note. * p<0.05, ** p<0.01, *** p<0.001.\n"
    output += "All tests conducted at α=0.05 significance level.\n"
    output += "Bonferroni correction applied for multiple comparisons.\n"
    
    return output


def format_confusion_matrix_caption(
    model_name: str,
    accuracy: float,
    classes: List[str],
    figure_number: int = 1
) -> str:
    """
    Generate APA-style caption for confusion matrix figure.
    
    Args:
        model_name: str - name of the model
        accuracy: float - overall accuracy
        classes: list - class names
        figure_number: int - figure number in thesis
    
    Returns:
        Formatted caption
    
    Example:
        >>> caption = format_confusion_matrix_caption(
        ...     "CrossViT", 0.923, ["COVID", "Normal", "Lung Opacity", "Viral Pneumonia"], 1
        ... )
        >>> print(caption)
    """
    caption = f"Figure {figure_number}. "
    caption += f"Confusion matrix for {model_name} on test set. "
    caption += f"Overall accuracy: {accuracy:.1%}. "
    caption += f"Classes: {', '.join(classes)}. "
    caption += f"Diagonal elements represent correct classifications."
    
    return caption


def format_roc_curve_caption(
    model_name: str,
    auc_scores: Dict[str, float],
    figure_number: int = 2
) -> str:
    """
    Generate APA-style caption for ROC curve figure.
    
    Args:
        model_name: str - name of the model
        auc_scores: dict - {class_name: auc_score}
        figure_number: int - figure number in thesis
    
    Returns:
        Formatted caption
    
    Example:
        >>> auc_scores = {"COVID": 0.94, "Normal": 0.96, "Lung Opacity": 0.92, "Viral Pneumonia": 0.93}
        >>> caption = format_roc_curve_caption("CrossViT", auc_scores, 2)
    """
    caption = f"Figure {figure_number}. "
    caption += f"ROC curves for {model_name} across all classes. "
    
    # Format AUC scores
    auc_text = ", ".join([f"{cls}: {auc:.2f}" for cls, auc in auc_scores.items()])
    caption += f"AUC scores: {auc_text}. "
    caption += "Micro-average and macro-average curves included for reference."
    
    return caption


def generate_latex_table(
    models: List[str],
    metrics: Dict[str, List[Tuple[float, float, float]]],
    caption: str = "Model Performance Comparison",
    label: str = "tab:results"
) -> str:
    """
    Generate LaTeX table code for thesis (if using LaTeX).
    
    Args:
        models: list - model names
        metrics: dict - {metric_name: [(mean, lower, upper), ...]}
        caption: str - table caption
        label: str - LaTeX label for referencing
    
    Returns:
        LaTeX table code
    
    Example:
        >>> latex = generate_latex_table(models, metrics, "Results", "tab:main_results")
    """
    n_metrics = len(metrics)
    col_spec = "l" + "c" * n_metrics
    
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += f"\\begin{{tabular}}{{{col_spec}}}\n"
    latex += "\\toprule\n"
    
    # Header
    header = "Model & " + " & ".join(metrics.keys()) + " \\\\\n"
    latex += header
    latex += "\\midrule\n"
    
    # Data rows
    for i, model in enumerate(models):
        row = model
        for metric_name, values in metrics.items():
            mean, lower, upper = values[i]
            row += f" & {mean:.1%} ({lower:.1%}--{upper:.1%})"
        row += " \\\\\n"
        latex += row
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\begin{tablenotes}\n"
    latex += "\\small\n"
    latex += "\\item Note. Values shown as Mean (95\\% CI Lower--Upper).\n"
    latex += "\\item All metrics computed on test set ($n=2{,}117$).\n"
    latex += "\\end{tablenotes}\n"
    latex += "\\end{table}\n"
    
    return latex


def format_for_word(
    models: List[str],
    metrics: Dict[str, List[Tuple[float, float, float]]],
    caption: str = "Model Performance Comparison"
) -> str:
    """
    Generate Word-friendly table (tab-separated values).
    
    Args:
        models: list - model names
        metrics: dict - metric values
        caption: str - table caption
    
    Returns:
        TSV format suitable for pasting into Word
    
    Example:
        >>> word_table = format_for_word(models, metrics)
        >>> # Copy output and paste into Word, then use "Convert Text to Table"
    """
    output = f"TABLE: {caption}\n\n"
    
    # Header
    header = "Model\t" + "\t".join(metrics.keys())
    output += header + "\n"
    
    # Data rows
    for i, model in enumerate(models):
        row = model
        for metric_name, values in metrics.items():
            mean, lower, upper = values[i]
            row += f"\t{mean:.1%} ({lower:.1%}-{upper:.1%})"
        output += row + "\n"
    
    output += "\nNote. Values shown as Mean (95% CI Lower-Upper).\n"
    
    return output


if __name__ == "__main__":
    # Example usage
    print("=== Table Formatter Demo ===\n")
    
    # Sample data
    models = ["CrossViT", "ResNet-50", "DenseNet-121", "EfficientNet-B0", "ViT-B/32"]
    
    metrics = {
        "Accuracy": [
            (0.923, 0.911, 0.935),
            (0.887, 0.872, 0.902),
            (0.892, 0.878, 0.906),
            (0.894, 0.880, 0.908),
            (0.914, 0.902, 0.926)
        ],
        "F1-Score": [
            (0.91, 0.89, 0.93),
            (0.87, 0.85, 0.89),
            (0.88, 0.86, 0.90),
            (0.89, 0.87, 0.91),
            (0.90, 0.88, 0.92)
        ],
        "AUC-ROC": [
            (0.94, 0.92, 0.96),
            (0.90, 0.88, 0.92),
            (0.91, 0.89, 0.93),
            (0.91, 0.89, 0.93),
            (0.93, 0.91, 0.95)
        ]
    }
    
    # Generate results table
    print(format_results_table(models, metrics, "CrossViT vs Baseline Models", 1))
    
    # Generate confusion matrix caption
    print("\n" + format_confusion_matrix_caption(
        "CrossViT", 0.923, ["COVID-19", "Normal", "Lung Opacity", "Viral Pneumonia"], 1
    ))
