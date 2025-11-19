# Phase 3: Analysis & Statistical Validation

**Duration:** Week 7-8 (Upcoming)
**Goal:** Statistical validation and deep analysis of results
**Status:** ⏸️ PENDING (Waiting for Phase 2 completion)

---

## 📋 Phase Overview

Phase 3 focuses on rigorous statistical analysis of the 30 training runs (6 models × 5 seeds), hypothesis testing, error analysis, and generation of publication-ready results for the thesis.

---

## 🎯 Objectives

### 1. Statistical Validation ⏸️
- Calculate 95% confidence intervals for all models
- Perform pairwise hypothesis testing
- Apply Bonferroni correction for multiple comparisons
- Calculate effect sizes (Cohen's d)
- Generate statistical significance tables

### 2. Model Comparison ⏸️
- Compare CrossViT vs each baseline (5 comparisons)
- Rank models by performance and consistency
- Analyze per-class performance
- Identify best model for each class

### 3. Error Analysis ⏸️
- Analyze misclassifications across all models
- Identify common failure patterns
- Visualize error cases
- Per-class confusion analysis

### 4. Ablation Studies ⏸️
- Test hypothesis H2-H4
- CLAHE vs no CLAHE comparison
- Augmentation impact analysis
- Dual-branch vs single-scale (CrossViT)

---

## 📊 Planned Notebooks

### 12_statistical_validation.ipynb ⏸️

**Purpose:** Rigorous statistical analysis of all results

**Tasks:**
- [ ] Load all 30 training runs from CSVs
- [ ] Calculate 95% CI for each model (bootstrap, 1000 iterations)
- [ ] Perform McNemar's test (CrossViT vs each baseline)
- [ ] Apply Bonferroni correction (α' = 0.01 for 5 comparisons)
- [ ] Calculate Cohen's d for effect sizes
- [ ] Generate APA-formatted results tables

**Expected Outputs:**
```
results/tables/
├── model_performance_summary.csv
├── confidence_intervals.csv
├── hypothesis_tests.csv
└── statistical_comparison_table.tex (for thesis)
```

**Key Research Questions:**
1. Is CrossViT significantly different from CNNs? (H₁)
2. Which differences are statistically significant?
3. What is the practical significance (effect size)?

---

### 13_error_analysis.ipynb ⏸️

**Purpose:** Deep analysis of model failures

**Tasks:**
- [ ] Load all confusion matrices (30 total)
- [ ] Identify most commonly misclassified classes
- [ ] Analyze per-class precision, recall, F1-score
- [ ] Visualize worst-performing images
- [ ] Compare error patterns across models

**Expected Outputs:**
```
results/figures/
├── aggregated_confusion_matrix.png
├── per_class_performance.png
├── error_analysis_heatmap.png
└── misclassified_examples.png
```

**Analysis Questions:**
1. Which classes are hardest to classify?
2. What are common confusion patterns? (e.g., COVID vs Viral Pneumonia)
3. Do different models make different errors?
4. Can we identify systematic biases?

---

### 14_ablation_studies.ipynb ⏸️

**Purpose:** Test remaining hypotheses and design choices

**Hypotheses to Test:**

**H₂: CLAHE Enhancement Improves Performance**
- Train ResNet-50 without CLAHE (5 seeds)
- Compare: CLAHE vs No CLAHE
- Expected: ≥2% improvement with CLAHE

**H₃: Conservative Augmentation Helps Generalization**
- Train ResNet-50 with no augmentation (5 seeds)
- Compare: Augmentation vs No Augmentation
- Expected: Better validation performance with augmentation

**H₄: Dual-Branch Architecture Improves CrossViT**
- Compare CrossViT (dual-branch) vs single-scale equivalent
- Analyze attention maps from both branches
- Expected: Dual-branch provides ≥5% improvement

**Expected Outputs:**
```
results/ablation/
├── clahe_comparison.csv
├── augmentation_comparison.csv
├── dual_branch_analysis.csv
└── attention_visualizations/
```

---

## 📋 Statistical Methods

### Confidence Intervals

**Method:** Bootstrap with 1000 iterations

```python
def calculate_ci(scores, confidence=0.95):
    """
    Calculate bootstrap confidence interval
    """
    n_bootstraps = 1000
    bootstrapped_means = []

    for _ in range(n_bootstraps):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrapped_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrapped_means, 100 * alpha / 2)
    upper = np.percentile(bootstrapped_means, 100 * (1 - alpha / 2))

    return np.mean(scores), lower, upper
```

**Output Format:**
```
Model: ResNet-50
Mean: 95.45%
95% CI: [94.88%, 96.02%]
```

---

### Hypothesis Testing

**Test:** McNemar's Test (paired binary classification test)

**Null Hypothesis (H₀):** CrossViT and baseline have equal error rates

**Alternative (H₁):** CrossViT and baseline have different error rates

**Significance Level:** α = 0.05 (Bonferroni corrected: α' = 0.01)

**Bonferroni Correction:**
- Number of comparisons: 5 (CrossViT vs 5 baselines)
- Adjusted α: 0.05 / 5 = 0.01

```python
from statsmodels.stats.contingency_tables import mcnemar

def compare_models(y_true, y_pred_a, y_pred_b):
    """
    McNemar's test for paired predictions
    """
    # Create contingency table
    table = [[sum((y_pred_a == y_true) & (y_pred_b == y_true)),
              sum((y_pred_a == y_true) & (y_pred_b != y_true))],
             [sum((y_pred_a != y_true) & (y_pred_b == y_true)),
              sum((y_pred_a != y_true) & (y_pred_b != y_true))]]

    result = mcnemar(table, exact=False, correction=True)
    return result.statistic, result.pvalue
```

---

### Effect Size

**Metric:** Cohen's d

```python
def cohens_d(scores_a, scores_b):
    """
    Calculate Cohen's d effect size
    """
    mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
    std_a, std_b = np.std(scores_a, ddof=1), np.std(scores_b, ddof=1)

    # Pooled standard deviation
    n_a, n_b = len(scores_a), len(scores_b)
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))

    d = (mean_a - mean_b) / pooled_std
    return d
```

**Interpretation:**
- |d| < 0.2: Small effect
- 0.2 ≤ |d| < 0.5: Small effect
- 0.5 ≤ |d| < 0.8: Medium effect
- |d| ≥ 0.8: Large effect

---

## 📊 Expected Results Tables (for Thesis Chapter 5)

### Table 5.1: Model Performance Summary

| Model | Mean Acc (%) | 95% CI | Std Dev | Best | Worst | Params |
|-------|-------------|--------|---------|------|-------|--------|
| ResNet-50 | 95.45 | [94.88, 96.02] | 0.57 | 96.03 | 94.66 | 23.5M |
| DenseNet-121 | 95.32 | [95.06, 95.58] | 0.26 | 95.65 | 94.99 | 7.0M |
| EfficientNet-B0 | 95.23 | [94.90, 95.56] | 0.33 | 95.65 | 94.80 | 4.0M |
| ViT-Base | TBD | TBD | TBD | TBD | TBD | 85.8M |
| Swin-Tiny | TBD | TBD | TBD | TBD | TBD | 27.5M |
| CrossViT-Tiny | 94.96 | [94.41, 95.51] | 0.55 | 95.65 | 94.33 | 7.0M |

---

### Table 5.2: Statistical Comparison (CrossViT vs Baselines)

| Comparison | Δ Acc | p-value | Significant? | Cohen's d | Interpretation |
|-----------|-------|---------|--------------|-----------|----------------|
| CrossViT vs ResNet-50 | -0.49% | TBD | TBD | TBD | TBD |
| CrossViT vs DenseNet-121 | -0.36% | TBD | TBD | TBD | TBD |
| CrossViT vs EfficientNet | -0.27% | TBD | TBD | TBD | TBD |
| CrossViT vs ViT-Base | TBD | TBD | TBD | TBD | TBD |
| CrossViT vs Swin-Tiny | TBD | TBD | TBD | TBD | TBD |

**Bonferroni-corrected α:** 0.01

---

### Table 5.3: Per-Class Performance (Best Model per Class)

| Class | Best Model | Precision | Recall | F1-Score | Support |
|-------|-----------|-----------|--------|----------|---------|
| COVID | TBD | TBD | TBD | TBD | 361 |
| Normal | TBD | TBD | TBD | TBD | 1,019 |
| Lung Opacity | TBD | TBD | TBD | TBD | 601 |
| Viral Pneumonia | TBD | TBD | TBD | TBD | 136 |

---

## 🔬 Preliminary Findings (from Phase 2)

### Finding 1: CNN Superiority

**Observation:** All CNN baselines outperformed CrossViT

**Possible Explanations:**
1. **Inductive Bias:** CNNs have translation invariance built-in
2. **Local Features:** Medical images have localized patterns (lesions, opacities)
3. **Data Efficiency:** Transformers typically need larger datasets
4. **Architecture Overhead:** CrossViT complexity doesn't translate to better performance
5. **Pre-training Domain:** ImageNet pre-training may favor CNNs for X-rays

**Discussion for Thesis:**
- Not a "failure" but a valid research finding
- Contributes to understanding when transformers are/aren't beneficial
- Practical implication: CNNs are sufficient for this task

---

### Finding 2: Consistency Matters

**Observation:** DenseNet-121 has lowest variance (±0.26%)

**Implication:**
- For clinical deployment, consistency is crucial
- DenseNet-121 most reliable despite not highest mean
- Lower risk of poor performance on new data

---

### Finding 3: Diminishing Returns

**Observation:** EfficientNet (4M params) nearly matches ResNet-50 (23.5M params)

**Implication:**
- Larger models don't guarantee better performance
- Model efficiency important for deployment
- ViT-Base (85.8M params) may not outperform smaller models

---

## 📁 Expected Files Generated in Phase 3

### Statistical Analysis:
```
results/statistical_analysis/
├── confidence_intervals.csv
├── hypothesis_tests.csv
├── effect_sizes.csv
├── per_class_metrics.csv
└── model_rankings.csv
```

### Figures:
```
results/figures/phase3/
├── model_comparison_boxplot.png
├── confidence_interval_plot.png
├── per_class_performance_radar.png
├── confusion_matrix_comparison.png
├── error_distribution.png
└── roc_curves_all_models.png
```

### Ablation Results:
```
results/ablation/
├── clahe_comparison_results.csv
├── augmentation_results.csv
├── dual_branch_results.csv
└── attention_visualizations/
```

### Thesis Tables (LaTeX format):
```
results/thesis_tables/
├── table_5_1_performance_summary.tex
├── table_5_2_statistical_comparison.tex
├── table_5_3_per_class_performance.tex
└── table_5_4_ablation_results.tex
```

---

## ✅ Phase 3 Success Criteria

- [ ] 95% CI calculated for all 6 models
- [ ] Hypothesis testing completed (McNemar's test)
- [ ] Bonferroni correction applied
- [ ] Effect sizes calculated (Cohen's d)
- [ ] Per-class analysis completed
- [ ] Error analysis with visualizations
- [ ] Ablation studies for H2, H3, H4 completed
- [ ] All tables formatted for thesis (APA 7th edition)
- [ ] All figures publication-ready (300 DPI)

---

## ⏱️ Estimated Timeline

**Week 7:**
- Day 1-2: Notebook 12 (Statistical Validation)
- Day 3-4: Notebook 13 (Error Analysis)
- Day 5-6: Start Notebook 14 (Ablation Studies)
- Day 7: Review and refine

**Week 8:**
- Day 1-3: Complete ablation studies
- Day 4-5: Generate all thesis tables/figures
- Day 6-7: Write analysis sections for Chapter 5

**Total Time:** ~2 weeks

---

## 📚 References for Phase 3

- Dietterich (1998) - Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms
- Demšar (2006) - Statistical Comparisons of Classifiers over Multiple Data Sets
- McNemar (1947) - Note on the Sampling Error of the Difference Between Correlated Proportions
- Cohen (1988) - Statistical Power Analysis for the Behavioral Sciences
- American Psychological Association (2020) - Publication Manual (7th ed.)

---

**Phase 3 Start Date:** TBD (After Phase 2 completes)
**Phase 3 Status:** ⏸️ PENDING
**Prerequisites:** All 30 training runs complete (currently 20/30)
**Next Phase:** Phase 4 - Documentation & Deployment
