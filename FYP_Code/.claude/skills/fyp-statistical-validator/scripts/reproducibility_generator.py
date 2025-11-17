#!/usr/bin/env python3
"""
Reproducibility Statement Generator for TAR UMT Thesis
Generates standardized experimental details and statistical methodology sections
"""

from typing import Dict, List, Optional
import platform
import torch


def generate_reproducibility_statement(
    random_seeds: List[int],
    n_runs: int,
    dataset_info: Dict,
    model_config: Dict,
    training_config: Dict,
    hardware_info: Optional[Dict] = None
) -> str:
    """
    Generate comprehensive reproducibility statement for thesis Chapter 4.
    
    This statement should be included in the "Experimental Setup" section
    to ensure other researchers can reproduce your results.
    
    Args:
        random_seeds: list - random seeds used (e.g., [42, 123, 456, 789, 101112])
        n_runs: int - number of independent runs
        dataset_info: dict - dataset details
        model_config: dict - model architecture details
        training_config: dict - training hyperparameters
        hardware_info: dict - hardware specifications (optional)
    
    Returns:
        Formatted reproducibility statement
    
    Example:
        >>> statement = generate_reproducibility_statement(
        ...     random_seeds=[42, 123, 456, 789, 101112],
        ...     n_runs=5,
        ...     dataset_info={
        ...         "name": "COVID-19 Radiography Database",
        ...         "train_size": 16932,
        ...         "val_size": 2116,
        ...         "test_size": 2117,
        ...         "n_classes": 4
        ...     },
        ...     model_config={
        ...         "name": "CrossViT-Tiny",
        ...         "input_size": "240x240",
        ...         "parameters": "7M"
        ...     },
        ...     training_config={
        ...         "epochs": 50,
        ...         "batch_size": 16,
        ...         "learning_rate": 5e-5,
        ...         "optimizer": "AdamW"
        ...     }
        ... )
    """
    statement = "\n" + "="*80 + "\n"
    statement += "REPRODUCIBILITY STATEMENT\n"
    statement += "="*80 + "\n\n"
    
    # 1. Random Seeds and Multiple Runs
    statement += "1. RANDOM SEEDS AND MULTIPLE RUNS\n"
    statement += "-" * 80 + "\n"
    statement += f"To ensure statistical validity and account for training variability, "
    statement += f"all experiments were conducted with {n_runs} independent runs using "
    statement += f"different random seeds: {random_seeds}. These seeds control:\n"
    statement += "  - Weight initialization\n"
    statement += "  - Data augmentation randomness\n"
    statement += "  - Training/validation/test split (with stratification)\n"
    statement += "  - Dropout mask generation during training\n\n"
    statement += f"All reported metrics represent the mean across {n_runs} runs with "
    statement += "95% confidence intervals calculated using the t-distribution.\n\n"
    
    # 2. Dataset Configuration
    statement += "2. DATASET CONFIGURATION\n"
    statement += "-" * 80 + "\n"
    statement += f"Dataset: {dataset_info['name']}\n"
    statement += f"  - Training set: {dataset_info['train_size']:,} images\n"
    statement += f"  - Validation set: {dataset_info['val_size']:,} images\n"
    statement += f"  - Test set: {dataset_info['test_size']:,} images\n"
    statement += f"  - Number of classes: {dataset_info['n_classes']}\n"
    statement += f"  - Split strategy: Stratified random sampling (80/10/10)\n\n"
    
    # 3. Model Configuration
    statement += "3. MODEL CONFIGURATION\n"
    statement += "-" * 80 + "\n"
    statement += f"Architecture: {model_config['name']}\n"
    statement += f"  - Input size: {model_config['input_size']} pixels\n"
    statement += f"  - Total parameters: {model_config['parameters']}\n"
    statement += f"  - Pre-training: ImageNet-21k\n"
    statement += f"  - Fine-tuning: Full model (all layers trainable)\n\n"
    
    # 4. Training Configuration
    statement += "4. TRAINING CONFIGURATION\n"
    statement += "-" * 80 + "\n"
    statement += f"  - Epochs: {training_config['epochs']}\n"
    statement += f"  - Batch size: {training_config['batch_size']}\n"
    statement += f"  - Learning rate: {training_config['learning_rate']}\n"
    statement += f"  - Optimizer: {training_config['optimizer']}\n"
    statement += f"  - Weight decay: {training_config.get('weight_decay', 0.05)}\n"
    statement += f"  - Learning rate scheduler: {training_config.get('scheduler', 'CosineAnnealingWarmRestarts')}\n"
    statement += f"  - Early stopping: Enabled (patience={training_config.get('early_stopping_patience', 15)})\n"
    statement += f"  - Mixed precision: {'Enabled (FP16)' if training_config.get('mixed_precision', True) else 'Disabled'}\n\n"
    
    # 5. Hardware Information (if provided)
    if hardware_info:
        statement += "5. HARDWARE CONFIGURATION\n"
        statement += "-" * 80 + "\n"
        statement += f"  - GPU: {hardware_info.get('gpu', 'N/A')}\n"
        statement += f"  - VRAM: {hardware_info.get('vram', 'N/A')}\n"
        statement += f"  - CPU: {hardware_info.get('cpu', 'N/A')}\n"
        statement += f"  - RAM: {hardware_info.get('ram', 'N/A')}\n"
        statement += f"  - OS: {hardware_info.get('os', platform.system())}\n\n"
    
    # 6. Statistical Methodology
    statement += "6. STATISTICAL METHODOLOGY\n"
    statement += "-" * 80 + "\n"
    statement += "All performance metrics are reported with 95% confidence intervals (CI) "
    statement += "calculated using the t-distribution for small sample sizes. The formula "
    statement += "used is:\n\n"
    statement += "    CI = mean ± t(α/2, df) × SE\n\n"
    statement += "where:\n"
    statement += "  - mean = average metric across n runs\n"
    statement += f"  - t(α/2, df) = critical value from t-distribution (df={n_runs-1})\n"
    statement += "  - SE = standard error = std / √n\n"
    statement += "  - α = 0.05 (95% confidence level)\n\n"
    statement += "Hypothesis testing was performed using:\n"
    statement += "  - McNemar's test: For single-run model comparisons\n"
    statement += "  - Paired t-test: For multi-run model comparisons\n"
    statement += "  - Bonferroni correction: For multiple comparison adjustment\n"
    statement += "  - Significance level: α = 0.05\n\n"
    
    # 7. Software Dependencies
    statement += "7. SOFTWARE DEPENDENCIES\n"
    statement += "-" * 80 + "\n"
    statement += f"  - Python: {platform.python_version()}\n"
    statement += f"  - PyTorch: {torch.__version__}\n"
    statement += "  - timm: 0.9.0+ (for CrossViT implementation)\n"
    statement += "  - scikit-learn: 1.3.0+ (for metrics)\n"
    statement += "  - NumPy: 1.24.0+\n"
    statement += "  - SciPy: 1.11.0+ (for statistical tests)\n\n"
    
    statement += "="*80 + "\n"
    statement += "Note: This reproducibility statement should be included in Chapter 4 "
    statement += "(Research Design)\n"
    statement += "of the thesis to ensure experimental transparency and enable result "
    statement += "replication.\n"
    statement += "="*80 + "\n"
    
    return statement


def generate_statistical_methods_section() -> str:
    """
    Generate the "Statistical Analysis Methods" subsection for Chapter 4.
    
    This provides the theoretical foundation for statistical validation
    used in your thesis.
    
    Returns:
        Formatted section text with citations
    """
    section = "\n"
    section += "4.X STATISTICAL ANALYSIS METHODS\n"
    section += "="*80 + "\n\n"
    
    section += "To ensure rigorous evaluation of model performance, this study employs "
    section += "multiple statistical validation techniques consistent with best practices "
    section += "in machine learning research (Dietterich, 1998; Demšar, 2006).\n\n"
    
    section += "4.X.1 Confidence Interval Estimation\n"
    section += "-"*80 + "\n\n"
    section += "All performance metrics are reported with 95% confidence intervals (CI) "
    section += "to quantify uncertainty in model performance estimates. For experiments "
    section += "with multiple independent runs (n=5), confidence intervals are calculated "
    section += "using the t-distribution:\n\n"
    section += "$$CI = \\bar{x} \\pm t_{\\alpha/2, df} \\times \\frac{s}{\\sqrt{n}}$$  ...(1)\n\n"
    section += "where $\\bar{x}$ is the sample mean, $t_{\\alpha/2, df}$ is the critical "
    section += "value from the t-distribution with degrees of freedom $df=n-1$, $s$ is "
    section += "the sample standard deviation, and $n$ is the number of runs.\n\n"
    section += "For single test set evaluations, the normal approximation method is used:\n\n"
    section += "$$CI = p \\pm z_{\\alpha/2} \\times \\sqrt{\\frac{p(1-p)}{N}}$$  ...(2)\n\n"
    section += "where $p$ is the observed accuracy and $N$ is the test set size.\n\n"
    
    section += "4.X.2 Hypothesis Testing\n"
    section += "-"*80 + "\n\n"
    section += "To assess whether observed performance differences between models are "
    section += "statistically significant, this study employs the following hypothesis tests:\n\n"
    
    section += "McNemar's Test (Dietterich, 1998): Used for comparing two classifiers on "
    section += "the same test set by analyzing disagreement patterns. The test statistic is:\n\n"
    section += "$$\\chi^2 = \\frac{(b-c)^2}{b+c}$$  ...(3)\n\n"
    section += "where $b$ represents cases where model 1 is correct and model 2 is wrong, "
    section += "and $c$ represents the opposite case.\n\n"
    
    section += "Paired t-test: Used for comparing mean performance across multiple "
    section += "independent runs with matched random seeds. The null hypothesis states "
    section += "that the mean difference between paired observations is zero:\n\n"
    section += "$$H_0: \\mu_d = 0, \\quad H_1: \\mu_d \\neq 0$$  ...(4)\n\n"
    
    section += "Bonferroni Correction: Applied when conducting multiple pairwise "
    section += "comparisons to control the family-wise error rate. The adjusted "
    section += "significance level is:\n\n"
    section += "$$\\alpha_{adj} = \\frac{\\alpha}{m}$$  ...(5)\n\n"
    section += "where $\\alpha$ is the desired family-wise error rate (0.05) and $m$ "
    section += "is the number of comparisons.\n\n"
    
    section += "All hypothesis tests are conducted at a significance level of $\\alpha=0.05$. "
    section += "Results with $p<0.05$ are considered statistically significant and marked "
    section += "with *, $p<0.01$ with **, and $p<0.001$ with ***.\n\n"
    
    section += "="*80 + "\n"
    section += "REQUIRED CITATIONS:\n"
    section += "-"*80 + "\n"
    section += "Demšar, J. (2006). Statistical comparisons of classifiers over multiple "
    section += "data sets. Journal of Machine Learning Research, 7, 1-30.\n\n"
    section += "Dietterich, T. G. (1998). Approximate statistical tests for comparing "
    section += "supervised classification learning algorithms. Neural Computation, "
    section += "10(7), 1895-1923. https://doi.org/10.1162/089976698300017197\n"
    section += "="*80 + "\n"
    
    return section


def generate_results_reporting_template(
    model_name: str,
    accuracy_mean: float,
    accuracy_ci: Tuple[float, float],
    baseline_name: str,
    baseline_mean: float,
    p_value: float
) -> str:
    """
    Generate standardized results reporting text for Chapter 5.
    
    Args:
        model_name: str - your model name (e.g., "CrossViT")
        accuracy_mean: float - mean accuracy
        accuracy_ci: tuple - (lower, upper) confidence interval
        baseline_name: str - baseline model name (e.g., "ResNet-50")
        baseline_mean: float - baseline accuracy
        p_value: float - statistical test p-value
    
    Returns:
        Formatted results text following TAR UMT standards
    
    Example:
        >>> text = generate_results_reporting_template(
        ...     "CrossViT", 0.923, (0.911, 0.935),
        ...     "ResNet-50", 0.887, 0.0001
        ... )
    """
    # Determine significance marker
    if p_value < 0.001:
        sig = "***"
        sig_text = "highly significant (p<0.001)"
    elif p_value < 0.01:
        sig = "**"
        sig_text = "very significant (p<0.01)"
    elif p_value < 0.05:
        sig = "*"
        sig_text = "significant (p<0.05)"
    else:
        sig = ""
        sig_text = "not statistically significant (p≥0.05)"
    
    lower, upper = accuracy_ci
    improvement = ((accuracy_mean - baseline_mean) / baseline_mean) * 100
    
    text = (f"The {model_name} model achieved a classification accuracy of "
            f"{accuracy_mean:.1%} (95% CI: {lower:.1%}-{upper:.1%}) on the test set, "
            f"significantly outperforming the {baseline_name} baseline which attained "
            f"{baseline_mean:.1%} accuracy. The difference is {sig_text}{sig}, "
            f"representing a {improvement:.1f}% relative improvement. Statistical "
            f"significance was assessed using McNemar's test for matched predictions "
            f"($\\chi^2$, p={p_value:.4f}).")
    
    return text


if __name__ == "__main__":
    # Example usage
    print("=== Reproducibility Statement Generator Demo ===\n")
    
    # Generate full reproducibility statement
    statement = generate_reproducibility_statement(
        random_seeds=[42, 123, 456, 789, 101112],
        n_runs=5,
        dataset_info={
            "name": "COVID-19 Radiography Database",
            "train_size": 16932,
            "val_size": 2116,
            "test_size": 2117,
            "n_classes": 4
        },
        model_config={
            "name": "CrossViT-Tiny",
            "input_size": "240x240",
            "parameters": "7M"
        },
        training_config={
            "epochs": 50,
            "batch_size": 16,
            "learning_rate": 5e-5,
            "optimizer": "AdamW",
            "weight_decay": 0.05,
            "scheduler": "CosineAnnealingWarmRestarts",
            "early_stopping_patience": 15,
            "mixed_precision": True
        },
        hardware_info={
            "gpu": "NVIDIA RTX 4060",
            "vram": "8GB",
            "cpu": "AMD Ryzen 7",
            "ram": "32GB",
            "os": "Windows 11"
        }
    )
    
    print(statement)
    
    # Generate statistical methods section
    print("\n" + generate_statistical_methods_section())
    
    # Generate results reporting example
    print("\n" + "="*80)
    print("EXAMPLE RESULTS REPORTING:")
    print("="*80 + "\n")
    results_text = generate_results_reporting_template(
        "CrossViT", 0.923, (0.911, 0.935),
        "ResNet-50", 0.887, 0.0001
    )
    print(results_text)
