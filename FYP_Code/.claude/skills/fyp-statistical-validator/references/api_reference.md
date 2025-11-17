# Statistical Validation API Reference

This document provides detailed API documentation for all statistical validation functions in the fyp-statistical-validator skill.

---

## TABLE OF CONTENTS

1. [Confidence Intervals](#confidence-intervals)
2. [Hypothesis Testing](#hypothesis-testing)
3. [Table Formatting](#table-formatting)
4. [Reproducibility](#reproducibility)

---

## CONFIDENCE INTERVALS

### When to Use Each Method

| Method | Best For | Sample Size | Assumptions |
|--------|----------|-------------|-------------|
| Normal Approximation | Single train/test split | n > 30 | Binomial distribution |
| Bootstrap | Robust estimates | Any | Minimal assumptions |
| Multi-Seed | Multiple runs | 5+ runs | Independent runs |

### Function Reference

#### `normal_approximation_ci(accuracy, n_samples, confidence=0.95)`

**Purpose:** Quick CI estimate from single model evaluation

**Parameters:**
- `accuracy` (float): Model accuracy between 0-1
- `n_samples` (int): Test set size
- `confidence` (float): Confidence level (default 0.95)

**Returns:** `(mean, lower_bound, upper_bound)`

**Example:**
```python
from scripts.confidence_intervals import normal_approximation_ci

mean, lower, upper = normal_approximation_ci(0.923, 2117)
print(f"{mean:.1%} (95% CI: {lower:.1%} - {upper:.1%})")
# Output: 92.3% (95% CI: 91.1% - 93.5%)
```

**When to Use:**
- Single train/test split only
- Need quick estimate
- Test set size > 30

**Limitations:**
- Assumes binomial distribution
- Less robust than bootstrap
- May underestimate variance

---

#### `bootstrap_ci(y_true, y_pred, n_iterations=1000, confidence=0.95, random_state=42)`

**Purpose:** Robust CI estimation through resampling

**Parameters:**
- `y_true` (array): True labels
- `y_pred` (array): Predicted labels
- `n_iterations` (int): Number of bootstrap samples (default 1000)
- `confidence` (float): Confidence level (default 0.95)
- `random_state` (int): Random seed for reproducibility

**Returns:** `(mean, lower_bound, upper_bound)`

**Example:**
```python
from scripts.confidence_intervals import bootstrap_ci
import numpy as np

y_true = np.array([0, 1, 1, 0, 1, ...])  # True labels
y_pred = np.array([0, 1, 1, 1, 1, ...])  # Predictions

mean, lower, upper = bootstrap_ci(y_true, y_pred, n_iterations=1000)
print(f"Bootstrap: {mean:.1%} (95% CI: {lower:.1%} - {upper:.1%})")
```

**When to Use:**
- Non-normal distributions
- Small sample sizes
- Need robust estimates
- Distribution shape unknown

**Advantages:**
- Minimal assumptions
- Works with any distribution
- More accurate for small samples

**Computational Cost:**
- 1000 iterations recommended
- Takes ~1-2 seconds

---

#### `multi_seed_ci(accuracies, confidence=0.95)`

**Purpose:** CI from multiple independent training runs

**Parameters:**
- `accuracies` (list): Accuracy from each seed (min 5 recommended)
- `confidence` (float): Confidence level (default 0.95)

**Returns:** `(mean, lower_bound, upper_bound, std_dev)`

**Example:**
```python
from scripts.confidence_intervals import multi_seed_ci

seeds_results = [0.921, 0.925, 0.920, 0.928, 0.919]
mean, lower, upper, std = multi_seed_ci(seeds_results)

print(f"{mean:.1%} ± {std:.1%} (95% CI: {lower:.1%} - {upper:.1%})")
# Output: 92.3% ± 1.2% (95% CI: 90.8% - 93.8%)
```

**When to Use:**
- Multiple training runs with different seeds
- Accounting for training variability
- Comparing models with multiple runs

**TAR UMT Requirement:**
- Minimum 5 seeds recommended
- Use t-distribution for small samples

**Best Practices:**
- Use same seeds for all models
- Report both mean±std and CI
- Include individual seed results

---

## HYPOTHESIS TESTING

### Test Selection Guide

| Comparison Type | Test to Use | When | Example |
|----------------|-------------|------|---------|
| Single run, same test set | McNemar's | CrossViT vs baseline (1 run each) | Single model evaluation |
| Multiple runs, paired seeds | Paired t-test | CrossViT vs baseline (5 seeds each) | Rigorous comparison |
| Multiple comparisons | Bonferroni | CrossViT vs 5 baselines | Prevent false positives |
| AUC comparison | DeLong | Comparing ROC curves | When using probabilistic outputs |

### Function Reference

#### `mcnemar_test(y_true, y_pred_model1, y_pred_model2)`

**Purpose:** Compare two classifiers on same test set

**Parameters:**
- `y_true` (array): True labels
- `y_pred_model1` (array): Predictions from model 1
- `y_pred_model2` (array): Predictions from model 2

**Returns:** `(statistic, p_value, interpretation)`

**Example:**
```python
from scripts.hypothesis_testing import mcnemar_test

stat, p, interp = mcnemar_test(y_true, y_pred_crossvit, y_pred_resnet)
print(f"McNemar: χ²={stat:.2f}, p={p:.4f}")
print(interp)
# Output: "Highly significant difference (p<0.001)***"
```

**Interpretation:**
- p < 0.001: *** (highly significant)
- p < 0.01: ** (very significant)
- p < 0.05: * (significant)
- p ≥ 0.05: Not significant

**When to Use:**
- Single run comparison
- Same test set for both models
- Binary or multi-class classification

**Null Hypothesis:** H₀: Both models have equal performance

---

#### `paired_ttest(scores_model1, scores_model2, alpha=0.05)`

**Purpose:** Compare models across multiple paired runs

**Parameters:**
- `scores_model1` (list): Scores from model 1 (e.g., [0.92, 0.93, ...])
- `scores_model2` (list): Scores from model 2 (must be same length)
- `alpha` (float): Significance level (default 0.05)

**Returns:** `(t_statistic, p_value, significant, interpretation)`

**Example:**
```python
from scripts.hypothesis_testing import paired_ttest

crossvit_seeds = [0.921, 0.925, 0.920, 0.928, 0.919]
resnet_seeds = [0.887, 0.892, 0.884, 0.895, 0.882]

t, p, sig, interp = paired_ttest(crossvit_seeds, resnet_seeds)
print(f"Paired t-test: t={t:.3f}, p={p:.4f}")
print(interp)
```

**Requirements:**
- Same number of runs for both models
- Runs paired by random seed
- Minimum 2 paired runs (5+ recommended)

**Null Hypothesis:** H₀: Mean difference = 0

---

#### `bonferroni_correction(p_values, alpha=0.05)`

**Purpose:** Adjust significance level for multiple comparisons

**Parameters:**
- `p_values` (list): P-values from multiple tests
- `alpha` (float): Desired family-wise error rate

**Returns:** `(significant_flags, adjusted_alpha, interpretation)`

**Example:**
```python
from scripts.hypothesis_testing import bonferroni_correction

# CrossViT vs 5 baselines
p_values = [0.0001, 0.0023, 0.0089, 0.0234, 0.0456]
sig_flags, adj_alpha, interp = bonferroni_correction(p_values)

print(f"Adjusted α: {adj_alpha:.4f}")  # 0.01 for 5 tests
print(f"Significant: {sum(sig_flags)}/{len(sig_flags)} tests")
```

**Formula:** α_adjusted = α / number_of_tests

**When to Use:**
- Comparing against multiple baselines
- Multiple pairwise comparisons
- Prevent inflated Type I error

**TAR UMT Requirement:**
- MUST use when comparing against 5 baselines
- Report both original and adjusted α

---

## TABLE FORMATTING

### Function Reference

#### `format_results_table(models, metrics, caption, table_number)`

**Purpose:** Generate APA-formatted results table for Chapter 5

**Parameters:**
- `models` (list): Model names ["CrossViT", "ResNet-50", ...]
- `metrics` (dict): {metric_name: [(mean, lower_ci, upper_ci), ...]}
- `caption` (str): Table caption
- `table_number` (int): Table number in thesis

**Example:**
```python
from scripts.table_formatter import format_results_table

models = ["CrossViT", "ResNet-50", "DenseNet-121"]
metrics = {
    "Accuracy": [
        (0.923, 0.911, 0.935),
        (0.887, 0.872, 0.902),
        (0.892, 0.878, 0.906)
    ],
    "F1-Score": [
        (0.91, 0.89, 0.93),
        (0.87, 0.85, 0.89),
        (0.88, 0.86, 0.90)
    ]
}

table = format_results_table(models, metrics, "CrossViT vs Baselines", 1)
print(table)
```

**Output Format:**
```
Table 1
CrossViT vs Baselines
================================================================================
      Model      Accuracy           F1-Score
   CrossViT  92.3% (91.1%-93.5%)  91.0% (89.0%-93.0%)
  ResNet-50  88.7% (87.2%-90.2%)  87.0% (85.0%-89.0%)
DenseNet-121  89.2% (87.8%-90.6%)  88.0% (86.0%-90.0%)
================================================================================
Note. Values shown as Mean (95% CI Lower-Upper).
All metrics computed on test set (n=2,117).
```

---

#### `format_confusion_matrix_caption(model_name, accuracy, classes, figure_number)`

**Purpose:** Generate APA-style figure caption

**Example:**
```python
from scripts.table_formatter import format_confusion_matrix_caption

caption = format_confusion_matrix_caption(
    "CrossViT", 
    0.923, 
    ["COVID-19", "Normal", "Lung Opacity", "Viral Pneumonia"],
    1
)
print(caption)
# Output: "Figure 1. Confusion matrix for CrossViT on test set. Overall accuracy: 92.3%. Classes: COVID-19, Normal, Lung Opacity, Viral Pneumonia. Diagonal elements represent correct classifications."
```

---

## REPRODUCIBILITY

### Function Reference

#### `generate_reproducibility_statement(...)`

**Purpose:** Generate comprehensive experimental setup documentation for Chapter 4

**Parameters:** (Multiple - see script for full list)
- `random_seeds`: List of random seeds used
- `n_runs`: Number of independent runs
- `dataset_info`: Dataset details dict
- `model_config`: Model architecture dict
- `training_config`: Training hyperparameters dict
- `hardware_info`: Hardware specifications dict (optional)

**Example:**
```python
from scripts.reproducibility_generator import generate_reproducibility_statement

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
        "optimizer": "AdamW"
    }
)

# Copy this entire statement into Chapter 4 "Experimental Setup" section
print(statement)
```

**What It Includes:**
1. Random seeds and multiple runs explanation
2. Dataset configuration details
3. Model architecture specifications
4. Training hyperparameters
5. Hardware configuration (if provided)
6. Statistical methodology explanation
7. Software dependencies

**Usage:** Copy entire output into Chapter 4 section 4.X "Reproducibility"

---

#### `generate_statistical_methods_section()`

**Purpose:** Generate theoretical foundation for statistical validation

**Returns:** Formatted section text with formulas and citations

**Example:**
```python
from scripts.reproducibility_generator import generate_statistical_methods_section

section = generate_statistical_methods_section()
# Copy this into Chapter 4 section 4.X "Statistical Analysis Methods"
print(section)
```

**What It Includes:**
- CI estimation theory with formulas
- Hypothesis testing methodology
- Test selection justification
- Required citations (Dietterich 1998, Demšar 2006)

---

#### `generate_results_reporting_template(...)`

**Purpose:** Generate standardized results text for Chapter 5

**Parameters:**
- `model_name`: Your model name
- `accuracy_mean`: Mean accuracy
- `accuracy_ci`: (lower, upper) tuple
- `baseline_name`: Baseline model name
- `baseline_mean`: Baseline accuracy
- `p_value`: Test p-value

**Example:**
```python
from scripts.reproducibility_generator import generate_results_reporting_template

text = generate_results_reporting_template(
    "CrossViT", 0.923, (0.911, 0.935),
    "ResNet-50", 0.887, 0.0001
)
print(text)
# Output: "The CrossViT model achieved a classification accuracy of 92.3% (95% CI: 91.1%-93.5%) on the test set, significantly outperforming the ResNet-50 baseline which attained 88.7% accuracy. The difference is highly significant (p<0.001)***, representing a 4.1% relative improvement..."
```

**Usage:** Use as template for writing Chapter 5 results paragraphs

---

## QUICK REFERENCE

### Complete Workflow Example

```python
# 1. Import all functions
from scripts.confidence_intervals import multi_seed_ci
from scripts.hypothesis_testing import mcnemar_test, paired_ttest, bonferroni_correction
from scripts.table_formatter import format_results_table
from scripts.reproducibility_generator import generate_reproducibility_statement

# 2. Calculate confidence intervals (5 seeds)
crossvit_seeds = [0.921, 0.925, 0.920, 0.928, 0.919]
mean, lower, upper, std = multi_seed_ci(crossvit_seeds)

# 3. Hypothesis testing
t, p, sig, interp = paired_ttest(crossvit_seeds, resnet_seeds)

# 4. Multiple comparison correction
p_values = [p_crossvit_resnet, p_crossvit_densenet, ...]
sig_flags, adj_alpha, interp = bonferroni_correction(p_values)

# 5. Generate results table
models = ["CrossViT", "ResNet-50", ...]
metrics = {"Accuracy": [...], "F1-Score": [...]}
table = format_results_table(models, metrics, "Results", 1)

# 6. Generate reproducibility statement
statement = generate_reproducibility_statement(...)

# 7. Copy outputs to thesis chapters
# - Table → Chapter 5
# - Statement → Chapter 4
```

---

## ERROR HANDLING

### Common Issues and Solutions

**Issue:** `ValueError: Need at least 2 runs for confidence interval`
**Solution:** Use at least 2 runs, preferably 5+

**Issue:** `ValueError: Both models must have same number of runs`
**Solution:** Ensure paired t-test has equal-length arrays

**Issue:** Bootstrap takes too long
**Solution:** Reduce n_iterations from 1000 to 100 for testing

**Issue:** CIs are too wide
**Solution:** Increase number of seeds (5 → 10) or test set size

---

## CITATIONS FOR THESIS

When using these methods in your thesis, cite:

```
Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation, 10(7), 1895-1923. https://doi.org/10.1162/089976698300017197

Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7, 1-30.

Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman and Hall/CRC. https://doi.org/10.1201/9780429246593
```

---

**END OF API REFERENCE**
