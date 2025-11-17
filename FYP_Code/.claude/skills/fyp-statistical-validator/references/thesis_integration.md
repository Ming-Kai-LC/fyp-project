# Thesis Integration Guide
## How to Use Statistical Validation in Each Chapter

This guide shows exactly where and how to integrate statistical validation into your TAR UMT FYP thesis.

---

## CHAPTER 1: INTRODUCTION

### Where to Include

**Section 1.4: Research Objectives and Hypothesis**

Must include formal hypotheses with statistical framework:

```markdown
**Hypotheses (REQUIRED)**

The primary null hypothesis (H₀) states that there is no significant difference in COVID-19 classification accuracy between CrossViT and traditional CNN baselines (p≥0.05). Conversely, the alternative hypothesis (H₁) proposes that CrossViT achieves significantly higher COVID-19 classification accuracy compared to CNN baselines (p<0.05).
```

### What to Use

- **Nothing from this skill needed in Chapter 1** - Just state your hypotheses clearly
- Chapter 1 sets up what you'll test, actual testing happens in Chapter 5

---

## CHAPTER 3: RESEARCH METHODOLOGY

### Where to Include

**Section 3.6: Statistical Validation Methods**

Explain the statistical tests you will use.

### What to Use

**Script:** `reproducibility_generator.py`
**Function:** `generate_statistical_methods_section()`

**Example:**
```python
from scripts.reproducibility_generator import generate_statistical_methods_section

section = generate_statistical_methods_section()
# Copy entire output into Chapter 3 Section 3.6
```

**What This Provides:**
- Theory behind confidence intervals
- Explanation of hypothesis tests (McNemar's, paired t-test, Bonferroni)
- Formulas with equation numbers
- Citations (Dietterich 1998, Demšar 2006)

**Thesis Template:**
```markdown
## 3.6 Statistical Validation Methods

To ensure rigorous evaluation of model performance, this study employs multiple statistical validation techniques...

[INSERT OUTPUT FROM generate_statistical_methods_section() HERE]

### 3.6.1 Confidence Interval Estimation
[Formula and explanation]

### 3.6.2 Hypothesis Testing
[Tests and justification]
```

---

## CHAPTER 4: RESEARCH DESIGN

### Where to Include

**Section 4.5: Experimental Setup and Reproducibility**

### What to Use

**Script:** `reproducibility_generator.py`
**Function:** `generate_reproducibility_statement(...)`

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
        "optimizer": "AdamW",
        "weight_decay": 0.05,
        "early_stopping_patience": 15,
        "mixed_precision": True
    },
    hardware_info={
        "gpu": "NVIDIA RTX 4060",
        "vram": "8GB",
        "cpu": "AMD Ryzen 7",
        "ram": "32GB"
    }
)

# Copy entire output into Chapter 4
print(statement)
```

**What This Provides:**
- Complete experimental setup documentation
- Random seeds used
- Dataset splits
- Training configuration
- Hardware specifications
- Statistical methodology explanation

**Thesis Template:**
```markdown
## 4.5 Experimental Setup

### 4.5.1 Reproducibility Statement

[INSERT OUTPUT FROM generate_reproducibility_statement() HERE]

To ensure experimental reproducibility, all experiments were conducted with 5 independent runs using random seeds: [42, 123, 456, 789, 101112]. These seeds control weight initialization, data augmentation, and training/validation/test splits...
```

---

## CHAPTER 5: RESULTS AND EVALUATION

This is the PRIMARY chapter where statistical validation is used extensively.

### Section 5.1: Model Performance Results

**Use:** Results tables with confidence intervals

**Script:** `table_formatter.py`
**Function:** `format_results_table(...)`

**Example:**
```python
from scripts.confidence_intervals import multi_seed_ci
from scripts.table_formatter import format_results_table

# Calculate CIs for each model
models = ["CrossViT", "ResNet-50", "DenseNet-121", "EfficientNet-B0", "ViT-B/32"]

# Accuracy results (5 seeds each)
crossvit_acc = [0.921, 0.925, 0.920, 0.928, 0.919]
resnet_acc = [0.887, 0.892, 0.884, 0.895, 0.882]
# ... etc for other models

# Calculate CIs
crossvit_ci = multi_seed_ci(crossvit_acc)[:3]  # (mean, lower, upper)
resnet_ci = multi_seed_ci(resnet_acc)[:3]
# ... etc

# Build metrics dict
metrics = {
    "Accuracy": [crossvit_ci, resnet_ci, ...],
    "F1-Score": [...],
    "AUC-ROC": [...]
}

# Generate table
table = format_results_table(
    models, 
    metrics, 
    "Classification Performance of CrossViT and Baseline Models",
    table_number=1
)
```

**Thesis Template:**
```markdown
## 5.1 Model Performance Results

The classification performance of CrossViT and five baseline models is presented in Table 1. All metrics are reported as mean values with 95% confidence intervals calculated from five independent runs using different random seeds.

[INSERT TABLE HERE]

The results demonstrate that CrossViT achieved the highest overall accuracy of...
```

---

### Section 5.2: Statistical Significance Testing

**Use:** Hypothesis testing to prove your H₁ hypothesis

**Scripts:** 
- `hypothesis_testing.py` (McNemar's, Paired t-test, Bonferroni)
- `table_formatter.py` (format_hypothesis_table)

**Example:**
```python
from scripts.hypothesis_testing import mcnemar_test, paired_ttest, bonferroni_correction
from scripts.table_formatter import format_hypothesis_table

# Prepare comparison pairs
comparisons = [
    ("CrossViT", "ResNet-50"),
    ("CrossViT", "DenseNet-121"),
    ("CrossViT", "EfficientNet-B0"),
    ("CrossViT", "ViT-B/32"),
    ("CrossViT", "Swin-Tiny")
]

# Run hypothesis tests for each pair
test_results = []
p_values = []

for model1, model2 in comparisons:
    # McNemar's test (single run)
    mcnemar_stat, mcnemar_p, _ = mcnemar_test(
        y_true, y_pred_crossvit, y_pred_baseline
    )
    
    # Paired t-test (5 seeds)
    t_stat, paired_p, _, _ = paired_ttest(
        crossvit_seeds, baseline_seeds
    )
    
    test_results.append({
        "mcnemar": (mcnemar_stat, mcnemar_p),
        "paired_t": (t_stat, paired_p)
    })
    p_values.append(paired_p)

# Apply Bonferroni correction
sig_flags, adj_alpha, interp = bonferroni_correction(p_values)

# Generate hypothesis testing table
table = format_hypothesis_table(comparisons, test_results, table_number=2)
print(table)
```

**Thesis Template:**
```markdown
## 5.2 Statistical Significance Testing

### 5.2.1 Hypothesis Testing Results

To assess whether the observed performance differences are statistically significant, pairwise comparisons were conducted between CrossViT and each baseline model using both McNemar's test (for single-run evaluations) and paired t-test (for multiple runs).

[INSERT HYPOTHESIS TESTING TABLE HERE]

Table 2 presents the statistical test results. All comparisons yielded p-values below the Bonferroni-corrected significance threshold of α=0.01 (adjusted from α=0.05 for 5 comparisons), indicating that CrossViT's superior performance is statistically significant across all baseline models.

### 5.2.2 Hypothesis Validation

Based on the statistical analysis:

**Null Hypothesis (H₀):** There is no significant difference in accuracy between CrossViT and baseline models (p≥0.05)

**Result:** H₀ is REJECTED (p<0.001 for all comparisons)

**Alternative Hypothesis (H₁):** CrossViT achieves significantly higher accuracy than baseline models (p<0.05)

**Result:** H₁ is ACCEPTED

The paired t-test results comparing CrossViT against ResNet-50 yielded t=4.23, p=0.013, indicating that CrossViT significantly outperforms ResNet-50 with a mean accuracy improvement of 3.6 percentage points (95% CI: 2.4-4.8 percentage points).
```

---

### Section 5.3: Detailed Results Analysis

**Use:** Formatted results reporting text

**Script:** `reproducibility_generator.py`
**Function:** `generate_results_reporting_template(...)`

**Example:**
```python
from scripts.reproducibility_generator import generate_results_reporting_template

# For each baseline comparison
text_resnet = generate_results_reporting_template(
    model_name="CrossViT",
    accuracy_mean=0.923,
    accuracy_ci=(0.911, 0.935),
    baseline_name="ResNet-50",
    baseline_mean=0.887,
    p_value=0.0001
)

text_densenet = generate_results_reporting_template(
    model_name="CrossViT",
    accuracy_mean=0.923,
    accuracy_ci=(0.911, 0.935),
    baseline_name="DenseNet-121",
    baseline_mean=0.892,
    p_value=0.0004
)

# Copy these formatted texts into your thesis
print(text_resnet)
print(text_densenet)
```

**Thesis Template:**
```markdown
## 5.3 Comparative Analysis

### 5.3.1 CrossViT vs ResNet-50

[INSERT generate_results_reporting_template OUTPUT HERE]

The CrossViT model achieved a classification accuracy of 92.3% (95% CI: 91.1%-93.5%) on the test set, significantly outperforming the ResNet-50 baseline which attained 88.7% accuracy...

### 5.3.2 CrossViT vs DenseNet-121

[INSERT NEXT TEMPLATE OUTPUT HERE]

Similarly, when compared against DenseNet-121 (89.2% accuracy), CrossViT demonstrated...
```

---

### Section 5.4: Figures with Proper Captions

**Use:** APA-formatted figure captions

**Script:** `table_formatter.py`
**Functions:** 
- `format_confusion_matrix_caption(...)`
- `format_roc_curve_caption(...)`

**Example:**
```python
from scripts.table_formatter import format_confusion_matrix_caption, format_roc_curve_caption

# Confusion matrix caption
cm_caption = format_confusion_matrix_caption(
    model_name="CrossViT",
    accuracy=0.923,
    classes=["COVID-19", "Normal", "Lung Opacity", "Viral Pneumonia"],
    figure_number=1
)

# ROC curve caption
roc_caption = format_roc_curve_caption(
    model_name="CrossViT",
    auc_scores={
        "COVID-19": 0.94,
        "Normal": 0.96,
        "Lung Opacity": 0.92,
        "Viral Pneumonia": 0.93
    },
    figure_number=2
)

print(cm_caption)
print(roc_caption)
```

**Thesis Template:**
```markdown
[CONFUSION MATRIX IMAGE HERE]

[INSERT format_confusion_matrix_caption() OUTPUT AS CAPTION]

Figure 1. Confusion matrix for CrossViT on test set. Overall accuracy: 92.3%. Classes: COVID-19, Normal, Lung Opacity, Viral Pneumonia. Diagonal elements represent correct classifications.

---

[ROC CURVES IMAGE HERE]

[INSERT format_roc_curve_caption() OUTPUT AS CAPTION]

Figure 2. ROC curves for CrossViT across all classes. AUC scores: COVID-19: 0.94, Normal: 0.96, Lung Opacity: 0.92, Viral Pneumonia: 0.93. Micro-average and macro-average curves included for reference.
```

---

## CHAPTER 6: FUTURE WORK

### Where to Mention

**Section 6.1: Limitations of Study**

Mention statistical limitations:

```markdown
While this study employed rigorous statistical validation with 5 independent runs and 95% confidence intervals, several statistical limitations exist:

1. Limited to 5 random seeds due to computational constraints (RTX 4060 8GB VRAM)
2. Single dataset validation (COVID-19 Radiography Database only)
3. Bonferroni correction may be conservative for 5 comparisons
```

---

## CHAPTER 7: CONCLUSION

### Where to Include

**Section 7.2: Achievement Summary**

Reference your statistical findings:

```markdown
## 7.2 Key Achievements

This study successfully:

1. **Proved Research Hypothesis:** Statistical analysis confirmed that CrossViT achieves significantly higher classification accuracy than all baseline models (p<0.001), rejecting the null hypothesis H₀.

2. **Rigorous Validation:** All performance metrics reported with 95% confidence intervals, following best practices in machine learning research (Dietterich, 1998; Demšar, 2006).

3. **Multiple Independent Runs:** Conducted 5 independent training runs with different random seeds to account for training variability and ensure result reliability.
```

---

## COMPLETE WORKFLOW SUMMARY

### Before Experiments

1. **Chapter 3:** Write statistical methods section using `generate_statistical_methods_section()`
2. **Chapter 4:** Document experimental setup using `generate_reproducibility_statement()`

### During Experiments

1. **Save Results:** Log accuracy/F1/AUC for each seed for each model
2. **Calculate CIs:** Use `multi_seed_ci()` for each model
3. **Run Tests:** Use `paired_ttest()` and `mcnemar_test()` for comparisons
4. **Apply Correction:** Use `bonferroni_correction()` for p-values

### After Experiments

1. **Chapter 5.1:** Generate results table with `format_results_table()`
2. **Chapter 5.2:** Generate hypothesis table with `format_hypothesis_table()`
3. **Chapter 5.3:** Write analysis paragraphs using `generate_results_reporting_template()`
4. **Chapter 5.4:** Add figure captions using `format_confusion_matrix_caption()` and `format_roc_curve_caption()`

---

## EXAMPLE PYTHON NOTEBOOK

Here's a complete example notebook structure:

```python
# Cell 1: Imports
from scripts.confidence_intervals import multi_seed_ci
from scripts.hypothesis_testing import paired_ttest, bonferroni_correction
from scripts.table_formatter import format_results_table
from scripts.reproducibility_generator import (
    generate_reproducibility_statement,
    generate_statistical_methods_section,
    generate_results_reporting_template
)

# Cell 2: Load Results
# Assuming you saved results from training
crossvit_seeds = [0.921, 0.925, 0.920, 0.928, 0.919]
resnet_seeds = [0.887, 0.892, 0.884, 0.895, 0.882]
# ... etc for other models

# Cell 3: Calculate Confidence Intervals
models_data = {
    "CrossViT": crossvit_seeds,
    "ResNet-50": resnet_seeds,
    # ... etc
}

ci_results = {}
for model_name, seeds in models_data.items():
    mean, lower, upper, std = multi_seed_ci(seeds)
    ci_results[model_name] = (mean, lower, upper)
    print(f"{model_name}: {mean:.1%} (95% CI: {lower:.1%}-{upper:.1%})")

# Cell 4: Hypothesis Testing
p_values = []
for baseline_name, baseline_seeds in [("ResNet-50", resnet_seeds), ...]:
    t, p, sig, interp = paired_ttest(crossvit_seeds, baseline_seeds)
    p_values.append(p)
    print(f"CrossViT vs {baseline_name}: {interp}")

# Cell 5: Bonferroni Correction
sig_flags, adj_alpha, interp = bonferroni_correction(p_values)
print(interp)

# Cell 6: Generate Tables
# ... code to generate tables

# Cell 7: Generate Thesis Content
# ... code to generate reproducibility statement, etc.

# Cell 8: Save Everything
# Save all outputs to files for copying into Word document
```

---

## CHECKLIST FOR TAR UMT COMPLIANCE

Before submitting Chapter 5, verify:

- [ ] All metrics have 95% confidence intervals
- [ ] Hypothesis testing performed for all baseline comparisons
- [ ] Bonferroni correction applied and reported
- [ ] Tables formatted in APA style
- [ ] Figure captions follow APA format
- [ ] Null hypothesis (H₀) explicitly stated and tested
- [ ] Alternative hypothesis (H₁) confirmed or rejected
- [ ] Significance levels (α=0.05) clearly stated
- [ ] Statistical test names and parameters provided
- [ ] Citations included (Dietterich 1998, Demšar 2006)
- [ ] All claims supported by statistical evidence

---

**END OF THESIS INTEGRATION GUIDE**
