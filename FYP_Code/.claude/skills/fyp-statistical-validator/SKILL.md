---
name: fyp-statistical-validator
description: Automate statistical validation, hypothesis testing, and confidence interval calculations required by TAR UMT thesis standards and top ML research conferences. Use when you need to (1) calculate 95% confidence intervals for model performance metrics, (2) perform hypothesis testing to compare models (McNemar's test, paired t-test, Bonferroni correction), (3) generate APA-formatted results tables for thesis chapters, (4) create reproducibility statements for experimental setup, (5) validate statistical significance of model improvements, or (6) format results according to academic publishing standards. Essential for FYP Chapter 4 (experimental setup documentation) and Chapter 5 (results with statistical validation).
---

# Statistical Validator for TAR UMT FYP

**Purpose:** Automate all statistical validation requirements for TAR UMT Data Science thesis, ensuring rigorous experimental validation and APA-compliant results reporting.

**Priority:** ⭐⭐⭐⭐⭐ CRITICAL - Every experiment MUST have statistical validation

---

## Quick Start

### The 3 Core Functions You Need

**1. Calculate Confidence Intervals**
```python
from scripts.confidence_intervals import multi_seed_ci

# For 5 training runs with different seeds
seeds_results = [0.921, 0.925, 0.920, 0.928, 0.919]
mean, lower, upper, std = multi_seed_ci(seeds_results)

# Output: 92.3% ± 1.2% (95% CI: 90.8% - 93.8%)
```

**2. Test Statistical Significance**
```python
from scripts.hypothesis_testing import paired_ttest

# Compare CrossViT vs ResNet-50 (5 seeds each)
crossvit_seeds = [0.921, 0.925, 0.920, 0.928, 0.919]
resnet_seeds = [0.887, 0.892, 0.884, 0.895, 0.882]

t, p, sig, interp = paired_ttest(crossvit_seeds, resnet_seeds)
# Output: "CrossViT significantly outperforms ResNet-50 (p<0.05)"
```

**3. Generate Results Table**
```python
from scripts.table_formatter import format_results_table

models = ["CrossViT", "ResNet-50", "DenseNet-121"]
metrics = {
    "Accuracy": [(0.923, 0.911, 0.935), (0.887, 0.872, 0.902), ...]
}
table = format_results_table(models, metrics, "Results", table_number=1)
# Output: APA-formatted table ready for thesis
```

---

## When to Use This Skill

### ✅ Always Use When:

1. **Reporting any model performance metric** → Need 95% CI
2. **Comparing models** → Need hypothesis testing
3. **Writing Chapter 5 results** → Need formatted tables
4. **Writing Chapter 4 setup** → Need reproducibility statement
5. **Claiming "significant improvement"** → Need p-value < 0.05
6. **Multiple model comparisons** → Need Bonferroni correction

### ❌ Don't Use When:

- Just exploring data (Chapter 2 literature review)
- Explaining methodology theory (covered by tar-umt-fyp-rds)
- Writing code for training (covered by crossvit-covid19-fyp)

---

## Core Capabilities

### 1. Confidence Interval Calculation

**Three methods available:**

| Method | When to Use | Example |
|--------|-------------|---------|
| **Normal Approximation** | Single run, quick estimate | `normal_approximation_ci(0.923, 2117)` |
| **Bootstrap** | Robust estimate, any distribution | `bootstrap_ci(y_true, y_pred)` |
| **Multi-Seed** | Multiple runs (RECOMMENDED) | `multi_seed_ci([0.92, 0.93, ...])` |

**TAR UMT Requirement:** ALWAYS report confidence intervals with metrics
- Format: "92.3% (95% CI: 91.1% - 93.5%)"
- Never report a single number without CI
- Use multi-seed method with 5+ runs for best practice

**Usage Pattern:**
```python
from scripts.confidence_intervals import multi_seed_ci, format_ci_result

# Calculate CI
mean, lower, upper, std = multi_seed_ci(accuracy_seeds)

# Format for thesis
result_text = format_ci_result(mean, lower, upper, std, "Accuracy")
# Output: "Accuracy: 92.3% ± 1.2% (95% CI: 90.8% - 93.8%)"
```

### 2. Hypothesis Testing Suite

**Three tests provided:**

#### A. McNemar's Test
**Use for:** Single run comparison, same test set
```python
from scripts.hypothesis_testing import mcnemar_test

stat, p, interp = mcnemar_test(y_true, y_pred_model1, y_pred_model2)
# Tests: Are prediction patterns significantly different?
```

#### B. Paired t-test
**Use for:** Multiple runs with matched seeds
```python
from scripts.hypothesis_testing import paired_ttest

t, p, sig, interp = paired_ttest(scores_model1, scores_model2)
# Tests: Is mean difference across runs significant?
```

#### C. Bonferroni Correction
**Use for:** REQUIRED when comparing against 5 baselines
```python
from scripts.hypothesis_testing import bonferroni_correction

p_values = [0.0001, 0.0023, 0.0089, 0.0234, 0.0456]
sig_flags, adj_alpha, interp = bonferroni_correction(p_values)
# Adjusts α: 0.05 / 5 comparisons = 0.01
```

**TAR UMT Requirement:** Must use Bonferroni when comparing against multiple baselines to control family-wise error rate.

### 3. APA Table Formatting

**Generate publication-ready tables:**

```python
from scripts.table_formatter import format_results_table

# Build metrics dictionary
metrics = {
    "Accuracy": [
        (0.923, 0.911, 0.935),  # CrossViT: (mean, lower_ci, upper_ci)
        (0.887, 0.872, 0.902),  # ResNet-50
        (0.892, 0.878, 0.906)   # DenseNet-121
    ],
    "F1-Score": [(0.91, 0.89, 0.93), (0.87, 0.85, 0.89), (0.88, 0.86, 0.90)],
    "AUC-ROC": [(0.94, 0.92, 0.96), (0.90, 0.88, 0.92), (0.91, 0.89, 0.93)]
}

# Generate table
table = format_results_table(
    models=["CrossViT", "ResNet-50", "DenseNet-121"],
    metrics=metrics,
    caption="Classification Performance Comparison",
    table_number=1
)

# Output: APA-formatted table with proper headers, footnotes, CI notation
```

**Also available:**
- `format_hypothesis_table()` → Statistical test results
- `format_confusion_matrix_caption()` → Figure captions
- `format_roc_curve_caption()` → ROC curve captions
- `generate_latex_table()` → LaTeX format (if needed)
- `format_for_word()` → TSV for Word import

### 4. Reproducibility Documentation

**Generate Chapter 4 experimental setup:**

```python
from scripts.reproducibility_generator import generate_reproducibility_statement

statement = generate_reproducibility_statement(
    random_seeds=[42, 123, 456, 789, 101112],
    n_runs=5,
    dataset_info={"name": "COVID-19 Radiography Database", ...},
    model_config={"name": "CrossViT-Tiny", ...},
    training_config={"epochs": 50, "batch_size": 16, ...},
    hardware_info={"gpu": "RTX 4060", "vram": "8GB", ...}
)

# Copy entire output into Chapter 4 section 4.5
```

**What it generates:**
1. Random seeds and multiple runs explanation
2. Dataset configuration details
3. Model architecture specifications
4. Training hyperparameters
5. Hardware specifications
6. Statistical methodology explanation
7. Software dependencies

**Also available:**
- `generate_statistical_methods_section()` → Theory for Chapter 3
- `generate_results_reporting_template()` → Formatted result paragraphs

---

## TAR UMT Thesis Integration

### Chapter 3: Research Methodology

**Section 3.6: Statistical Validation Methods**

Use: `generate_statistical_methods_section()`

Provides theory, formulas, and citations for CI estimation and hypothesis testing.

### Chapter 4: Research Design

**Section 4.5: Experimental Setup and Reproducibility**

Use: `generate_reproducibility_statement(...)`

Documents all experimental details for reproducibility.

### Chapter 5: Results and Evaluation

**Section 5.1: Model Performance**
- Use: `format_results_table()` for performance comparison
- All metrics MUST have 95% CI

**Section 5.2: Statistical Significance**
- Use: `paired_ttest()`, `bonferroni_correction()`
- Use: `format_hypothesis_table()` for test results
- MUST prove H₁ hypothesis with p<0.05

**Section 5.3: Detailed Analysis**
- Use: `generate_results_reporting_template()` for formatted paragraphs
- Format: "CrossViT achieved X% (95% CI: Y%-Z%), significantly outperforming baseline (p<0.05)"

**Section 5.4: Figures**
- Use: `format_confusion_matrix_caption()` for Figure 1
- Use: `format_roc_curve_caption()` for Figure 2

---

## Complete Workflow Example

### Step 1: After Training (Collect Results)

```python
# Save results from each seed
crossvit_acc = [0.921, 0.925, 0.920, 0.928, 0.919]
crossvit_f1 = [0.91, 0.92, 0.90, 0.93, 0.89]
crossvit_auc = [0.94, 0.95, 0.93, 0.96, 0.92]

# Same for all 5 baselines
resnet_acc = [0.887, 0.892, 0.884, 0.895, 0.882]
# ... etc
```

### Step 2: Calculate Confidence Intervals

```python
from scripts.confidence_intervals import multi_seed_ci

# For each model and metric
crossvit_acc_ci = multi_seed_ci(crossvit_acc)[:3]  # (mean, lower, upper)
crossvit_f1_ci = multi_seed_ci(crossvit_f1)[:3]
crossvit_auc_ci = multi_seed_ci(crossvit_auc)[:3]

# Repeat for all baselines
resnet_acc_ci = multi_seed_ci(resnet_acc)[:3]
# ... etc
```

### Step 3: Hypothesis Testing

```python
from scripts.hypothesis_testing import paired_ttest, bonferroni_correction

# Compare CrossViT against each baseline
p_values = []
for baseline_name, baseline_seeds in baselines.items():
    t, p, sig, interp = paired_ttest(crossvit_acc, baseline_seeds)
    p_values.append(p)
    print(f"CrossViT vs {baseline_name}: {interp}")

# Apply Bonferroni correction
sig_flags, adj_alpha, interp = bonferroni_correction(p_values)
print(f"After correction: {interp}")
```

### Step 4: Generate Tables

```python
from scripts.table_formatter import format_results_table

models = ["CrossViT", "ResNet-50", "DenseNet-121", "EfficientNet-B0", "ViT-B/32"]
metrics = {
    "Accuracy": [crossvit_acc_ci, resnet_acc_ci, densenet_acc_ci, ...],
    "F1-Score": [crossvit_f1_ci, resnet_f1_ci, densenet_f1_ci, ...],
    "AUC-ROC": [crossvit_auc_ci, resnet_auc_ci, densenet_auc_ci, ...]
}

table1 = format_results_table(models, metrics, "Model Performance Comparison", 1)
```

### Step 5: Generate Reproducibility Statement

```python
from scripts.reproducibility_generator import generate_reproducibility_statement

statement = generate_reproducibility_statement(
    random_seeds=[42, 123, 456, 789, 101112],
    n_runs=5,
    dataset_info={...},
    model_config={...},
    training_config={...},
    hardware_info={...}
)
```

### Step 6: Copy to Thesis

1. **Chapter 4** ← Reproducibility statement
2. **Chapter 5.1** ← Results table
3. **Chapter 5.2** ← Hypothesis testing results
4. **Chapter 5.3** ← Formatted analysis paragraphs

---

## Reference Documentation

For detailed API documentation, see:
- **[api_reference.md](references/api_reference.md)** - Complete function reference with examples
- **[thesis_integration.md](references/thesis_integration.md)** - Chapter-by-chapter integration guide

---

## Quick Reference Table

| Task | Function | Script |
|------|----------|--------|
| Calculate CI (5 seeds) | `multi_seed_ci()` | confidence_intervals.py |
| Calculate CI (single run) | `normal_approximation_ci()` | confidence_intervals.py |
| Calculate CI (robust) | `bootstrap_ci()` | confidence_intervals.py |
| Compare 2 models (single) | `mcnemar_test()` | hypothesis_testing.py |
| Compare 2 models (seeds) | `paired_ttest()` | hypothesis_testing.py |
| Multiple comparisons | `bonferroni_correction()` | hypothesis_testing.py |
| Results table | `format_results_table()` | table_formatter.py |
| Hypothesis table | `format_hypothesis_table()` | table_formatter.py |
| Figure caption | `format_confusion_matrix_caption()` | table_formatter.py |
| Reproducibility | `generate_reproducibility_statement()` | reproducibility_generator.py |
| Statistical methods | `generate_statistical_methods_section()` | reproducibility_generator.py |
| Results template | `generate_results_reporting_template()` | reproducibility_generator.py |

---

## Critical Reminders

### TAR UMT Requirements (MUST DO):

1. ✅ **Every metric MUST have 95% CI** - Never report single numbers
2. ✅ **Minimum 5 seeds required** - Use different random seeds for each run
3. ✅ **Hypothesis testing REQUIRED** - Must prove H₁ with p<0.05
4. ✅ **Bonferroni correction** - Apply when comparing against 5 baselines
5. ✅ **APA table format** - Use provided formatters for consistency
6. ✅ **Reproducibility statement** - Include in Chapter 4
7. ✅ **Significance markers** - Use *, **, *** for p-values

### Best Practices:

- Always use paired t-test for multi-seed comparisons
- Report both mean±std AND 95% CI
- Include all individual seed results in appendix
- State null hypothesis (H₀) explicitly before testing
- Use adj_alpha from Bonferroni, not original α=0.05

### Common Mistakes to Avoid:

❌ Reporting accuracy without confidence interval
❌ Claiming "better" without statistical test
❌ Using single run for final comparison
❌ Forgetting Bonferroni correction for multiple tests
❌ Not stating hypotheses in Chapter 1
❌ Missing reproducibility statement in Chapter 4

---

## Integration with Other Skills

**This skill works with:**

- **crossvit-covid19-fyp** → Technical specs for experiments
- **fyp-jupyter** → Running experiments that generate results to validate
- **fyp-chapter-bridge** → Converting validated results into thesis text
- **tar-umt-fyp-rds** → Understanding thesis structure requirements

**Typical workflow:**
1. Use **fyp-jupyter** to run experiments → Get results
2. Use **fyp-statistical-validator** (this skill) → Validate results statistically
3. Use **fyp-chapter-bridge** → Convert into thesis chapters
4. Use **tar-umt-academic-writing** → Add APA citations

---

## Testing the Skill

Run demo scripts to verify everything works:

```bash
# Test confidence intervals
python scripts/confidence_intervals.py

# Test hypothesis testing
python scripts/hypothesis_testing.py

# Test table formatting
python scripts/table_formatter.py

# Test reproducibility generator
python scripts/reproducibility_generator.py
```

Each script includes `if __name__ == "__main__"` blocks with usage examples.

---

## Required Citations

When using these methods in your thesis, cite:

```
Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation, 10(7), 1895-1923. https://doi.org/10.1162/089976698300017197

Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7, 1-30.

Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman and Hall/CRC. https://doi.org/10.1201/9780429246593
```

---

## Success Criteria

You know this skill is working when:

✅ All metrics in Chapter 5 have 95% CI
✅ Hypothesis testing proves H₁ with p<0.05
✅ Tables formatted in APA style
✅ Reproducibility statement in Chapter 4
✅ Bonferroni correction applied for 5 comparisons
✅ Results ready to copy-paste into Word document

---

## Summary

**This skill automates all statistical validation for TAR UMT FYP.**

**Three core functions:**
1. `multi_seed_ci()` → Calculate confidence intervals
2. `paired_ttest()` → Test statistical significance
3. `format_results_table()` → Generate APA tables

**Use in every thesis chapter that reports experimental results (Chapters 4-5).**

**Saves 20-30 hours of manual calculation and formatting during thesis writing.**

---

**For detailed documentation, see references/api_reference.md and references/thesis_integration.md**
