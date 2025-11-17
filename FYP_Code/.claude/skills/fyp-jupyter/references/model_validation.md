# Model Validation and Statistical Testing - FYP Requirements

This reference provides comprehensive guidance on statistical validation required for academic Final Year Projects, including confidence intervals, hypothesis testing, and effect sizes.

## Performance Metrics

### Classification Metrics

**Accuracy**
```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```
- **Use when**: Balanced classes
- **Interpretation**: Proportion of correct predictions
- **Range**: [0, 1], higher is better

**Precision, Recall, F1-Score**
```python
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```
- **Precision**: Of predicted positives, how many were actually positive
- **Recall**: Of actual positives, how many were correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Use when**: Imbalanced classes, different costs for FP and FN

**Confusion Matrix**
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualize
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Calculate sensitivity and specificity
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # True Positive Rate
specificity = tn / (tn + fp)  # True Negative Rate
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
```

**ROC-AUC**
```python
from sklearn.metrics import roc_auc_score, roc_curve

# For binary classification
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

print(f"ROC-AUC Score: {roc_auc:.4f}")
```
- **Use when**: Want to evaluate model across all classification thresholds
- **Range**: [0, 1], 0.5 = random, 1.0 = perfect

### Regression Metrics

**Mean Absolute Error (MAE)**
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")
```
- **Interpretation**: Average absolute difference between predictions and actual values
- **Units**: Same as target variable
- **Use when**: Want interpretable error in original units

**Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)**
```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
```
- **Interpretation**: Penalizes larger errors more heavily
- **Use when**: Want to penalize large errors

**R² Score (Coefficient of Determination)**
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
```
- **Interpretation**: Proportion of variance explained by the model
- **Range**: (-∞, 1], 1.0 = perfect, 0 = baseline mean predictor
- **Use when**: Want to know proportion of variance explained

**Mean Absolute Percentage Error (MAPE)**
```python
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape:.2f}%")
```
- **Interpretation**: Percentage error
- **Use when**: Want percentage-based error metric

## Confidence Intervals (REQUIRED FOR FYP)

### Bootstrap Confidence Intervals

**For any metric:**
```python
from sklearn.utils import resample
import numpy as np

def bootstrap_ci(y_true, y_pred, metric_func, n_iterations=1000, confidence=0.95):
    """
    Calculate bootstrap confidence interval for any metric.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    metric_func : callable
        Metric function (e.g., accuracy_score, f1_score)
    n_iterations : int
        Number of bootstrap samples
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
    --------
    mean : float
        Mean metric value
    lower : float
        Lower bound of CI
    upper : float
        Upper bound of CI
    """
    scores = []
    n_size = len(y_true)
    
    for i in range(n_iterations):
        # Resample with replacement
        indices = resample(range(n_size), n_samples=n_size)
        y_true_boot = [y_true[i] for i in indices]
        y_pred_boot = [y_pred[i] for i in indices]
        
        # Calculate metric
        score = metric_func(y_true_boot, y_pred_boot)
        scores.append(score)
    
    # Calculate CI
    alpha = (1 - confidence) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)
    mean = np.mean(scores)
    
    return mean, lower, upper

# Example usage
from sklearn.metrics import accuracy_score

mean_acc, lower_ci, upper_ci = bootstrap_ci(y_test, y_pred, accuracy_score)
print(f"Accuracy: {mean_acc:.4f}")
print(f"95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]")
```

### Cross-Validation Confidence Intervals

**Using cross-validation scores:**
```python
from sklearn.model_selection import cross_val_score
from scipy import stats

def cv_confidence_interval(model, X, y, cv=10, confidence=0.95):
    """Calculate CI from cross-validation scores."""
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    mean = np.mean(scores)
    n = len(scores)
    std_error = stats.sem(scores)
    margin = std_error * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return mean, mean - margin, mean + margin

# Example
mean_acc, lower_ci, upper_ci = cv_confidence_interval(model, X_train, y_train)
print(f"Mean CV Accuracy: {mean_acc:.4f}")
print(f"95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]")
```

### Reporting Template for FYP

```python
# Comprehensive performance report with CI
def generate_performance_report(model, X_train, y_train, X_test, y_test, cv=10):
    """Generate complete performance report with confidence intervals."""
    
    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    mean_cv, lower_cv, upper_cv = cv_confidence_interval(model, X_train, y_train, cv=cv)
    
    # Test set performance
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Bootstrap CI for test accuracy
    mean_test, lower_test, upper_test = bootstrap_ci(y_test, y_pred, accuracy_score)
    
    # Generate report
    report = {
        'Metric': ['CV Accuracy', 'Test Accuracy'],
        'Mean': [f"{mean_cv:.4f}", f"{test_accuracy:.4f}"],
        '95% CI Lower': [f"{lower_cv:.4f}", f"{lower_test:.4f}"],
        '95% CI Upper': [f"{upper_cv:.4f}", f"{upper_test:.4f}"],
        'CI Width': [f"{upper_cv - lower_cv:.4f}", f"{upper_test - lower_test:.4f}"]
    }
    
    report_df = pd.DataFrame(report)
    print("\n=== Performance Report ===")
    print(report_df.to_string(index=False))
    print(f"\nCV Scores: {cv_scores}")
    print(f"CV Std Dev: {np.std(cv_scores):.4f}")
    
    return report_df

# Usage
report_df = generate_performance_report(model, X_train, y_train, X_test, y_test)
report_df.to_csv('performance_report.csv', index=False)
```

## Hypothesis Testing

### Comparing Two Models (Paired t-test)

**When to use**: Compare performance of two models on the same dataset.

```python
from scipy.stats import ttest_rel, ttest_ind
from sklearn.model_selection import cross_val_score

# Get cross-validation scores for both models
model1_scores = cross_val_score(model1, X, y, cv=10, scoring='accuracy')
model2_scores = cross_val_score(model2, X, y, cv=10, scoring='accuracy')

# Paired t-test (same folds)
t_stat, p_value = ttest_rel(model1_scores, model2_scores)

print(f"Model 1 mean: {np.mean(model1_scores):.4f} ± {np.std(model1_scores):.4f}")
print(f"Model 2 mean: {np.mean(model2_scores):.4f} ± {np.std(model2_scores):.4f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print(f"✓ Statistically significant difference (p < {alpha})")
    if np.mean(model1_scores) > np.mean(model2_scores):
        print("  Model 1 is significantly better")
    else:
        print("  Model 2 is significantly better")
else:
    print(f"✗ No statistically significant difference (p >= {alpha})")
```

### Comparing with Baseline

**Essential for FYP**: Always compare your model with a baseline.

```python
from sklearn.dummy import DummyClassifier

# Create baseline (majority class classifier)
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)

# Get scores
baseline_scores = cross_val_score(baseline, X, y, cv=10, scoring='accuracy')
model_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')

# Statistical test
t_stat, p_value = ttest_rel(model_scores, baseline_scores)

print("=== Model vs. Baseline Comparison ===")
print(f"Baseline accuracy: {np.mean(baseline_scores):.4f} ± {np.std(baseline_scores):.4f}")
print(f"Model accuracy: {np.mean(model_scores):.4f} ± {np.std(model_scores):.4f}")
print(f"Improvement: {(np.mean(model_scores) - np.mean(baseline_scores)) * 100:.2f}%")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("✓ Model significantly outperforms baseline (p < 0.05)")
else:
    print("✗ No significant improvement over baseline (p >= 0.05)")
```

### Multiple Comparison Correction (Bonferroni)

**When comparing multiple models:**

```python
from scipy.stats import ttest_rel

models = [model1, model2, model3, model4]
model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4']

# Get scores for all models
all_scores = []
for model in models:
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    all_scores.append(scores)

# Pairwise comparisons
n_comparisons = len(models) * (len(models) - 1) // 2
alpha = 0.05
bonferroni_alpha = alpha / n_comparisons

print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.4f}")
print("\n=== Pairwise Comparisons ===")

for i in range(len(models)):
    for j in range(i + 1, len(models)):
        t_stat, p_value = ttest_rel(all_scores[i], all_scores[j])
        significant = "✓" if p_value < bonferroni_alpha else "✗"
        print(f"{model_names[i]} vs {model_names[j]}: p={p_value:.4f} {significant}")
```

## Effect Size

### Cohen's d

**Measures practical significance beyond statistical significance.**

```python
def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.
    
    Interpretation:
    - Small effect: d ≈ 0.2
    - Medium effect: d ≈ 0.5
    - Large effect: d ≈ 0.8
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    d = (mean1 - mean2) / pooled_std
    return d

# Example
d = cohens_d(model_scores, baseline_scores)
print(f"Cohen's d: {d:.4f}")

if abs(d) < 0.2:
    effect = "Negligible"
elif abs(d) < 0.5:
    effect = "Small"
elif abs(d) < 0.8:
    effect = "Medium"
else:
    effect = "Large"

print(f"Effect size: {effect}")
```

### Hedges' g (Correction for Small Samples)

```python
def hedges_g(group1, group2):
    """Calculate Hedges' g (corrected Cohen's d for small samples)."""
    d = cohens_d(group1, group2)
    n = len(group1) + len(group2)
    correction = 1 - (3 / (4 * n - 9))
    g = d * correction
    return g

g = hedges_g(model_scores, baseline_scores)
print(f"Hedges' g: {g:.4f}")
```

### Practical Significance vs Statistical Significance

```python
def evaluate_significance(model_scores, baseline_scores, alpha=0.05):
    """Evaluate both statistical and practical significance."""
    
    # Statistical significance
    t_stat, p_value = ttest_rel(model_scores, baseline_scores)
    statistically_significant = p_value < alpha
    
    # Practical significance
    effect_size = cohens_d(model_scores, baseline_scores)
    practically_significant = abs(effect_size) >= 0.5  # Medium or larger effect
    
    # Report
    print("=== Significance Evaluation ===")
    print(f"Statistical significance (p < {alpha}): {statistically_significant} (p = {p_value:.4f})")
    print(f"Effect size (Cohen's d): {effect_size:.4f}")
    print(f"Practical significance (|d| >= 0.5): {practically_significant}")
    
    # Interpretation matrix
    if statistically_significant and practically_significant:
        print("\n✓✓ Both statistically and practically significant")
        print("   → Strong evidence for model improvement")
    elif statistically_significant and not practically_significant:
        print("\n✓✗ Statistically significant but small effect")
        print("   → Weak evidence; improvement may not be meaningful")
    elif not statistically_significant and practically_significant:
        print("\n✗✓ Large effect but not statistically significant")
        print("   → May need more data to confirm")
    else:
        print("\n✗✗ Neither statistically nor practically significant")
        print("   → No evidence of meaningful improvement")
    
    return statistically_significant, practically_significant

evaluate_significance(model_scores, baseline_scores)
```

## Model Comparison Best Practices

### Comprehensive Comparison Framework

```python
def comprehensive_model_comparison(models, model_names, X, y, cv=10):
    """
    Comprehensive comparison of multiple models with statistical tests.
    """
    results = []
    all_scores = []
    
    print("=== Training and Evaluating Models ===")
    for name, model in zip(model_names, models):
        # Cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        all_scores.append(scores)
        
        # Calculate statistics
        mean = np.mean(scores)
        std = np.std(scores)
        ci_lower, ci_upper = np.percentile(scores, [2.5, 97.5])
        
        results.append({
            'Model': name,
            'Mean Accuracy': f"{mean:.4f}",
            'Std Dev': f"{std:.4f}",
            '95% CI': f"[{ci_lower:.4f}, {ci_upper:.4f}]"
        })
        
        print(f"{name}: {mean:.4f} ± {std:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    # Statistical comparisons
    print("\n=== Statistical Comparisons ===")
    best_idx = np.argmax([np.mean(scores) for scores in all_scores])
    
    for i, name in enumerate(model_names):
        if i != best_idx:
            t_stat, p_value = ttest_rel(all_scores[best_idx], all_scores[i])
            d = cohens_d(all_scores[best_idx], all_scores[i])
            
            print(f"\n{model_names[best_idx]} vs {name}:")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Cohen's d: {d:.4f}")
            
            if p_value < 0.05:
                print(f"  ✓ {model_names[best_idx]} significantly better")
            else:
                print(f"  ✗ No significant difference")
    
    return results_df, all_scores

# Example usage
models = [
    LogisticRegression(max_iter=1000),
    RandomForestClassifier(n_estimators=100, random_state=42),
    XGBClassifier(n_estimators=100, random_state=42)
]

model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']

results_df, all_scores = comprehensive_model_comparison(models, model_names, X, y)
results_df.to_csv('model_comparison.csv', index=False)
```

## Validation Strategies

### K-Fold Cross-Validation
```python
from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"CV Scores: {scores}")
print(f"Mean: {np.mean(scores):.4f}")
print(f"Std: {np.std(scores):.4f}")
```

### Stratified K-Fold (for Imbalanced Classes)
```python
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')
```

### Repeated K-Fold
```python
from sklearn.model_selection import RepeatedStratifiedKFold

rskfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scores = cross_val_score(model, X, y, cv=rskfold, scoring='accuracy')
```

## FYP Results Reporting Template

```python
def generate_fyp_results_report(model, baseline, X_train, y_train, X_test, y_test):
    """
    Generate comprehensive FYP-compliant results report.
    Includes: CV performance, test performance, CI, statistical tests, effect sizes.
    """
    
    print("=" * 60)
    print("FINAL YEAR PROJECT RESULTS REPORT")
    print("=" * 60)
    
    # 1. Cross-validation results
    print("\n1. CROSS-VALIDATION RESULTS (Training Set)")
    print("-" * 60)
    model_cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    baseline_cv_scores = cross_val_score(baseline, X_train, y_train, cv=10, scoring='accuracy')
    
    model_mean, model_lower, model_upper = cv_confidence_interval(model, X_train, y_train)
    baseline_mean, baseline_lower, baseline_upper = cv_confidence_interval(baseline, X_train, y_train)
    
    print(f"Proposed Model:")
    print(f"  Mean Accuracy: {model_mean:.4f}")
    print(f"  95% CI: [{model_lower:.4f}, {model_upper:.4f}]")
    print(f"  Std Dev: {np.std(model_cv_scores):.4f}")
    
    print(f"\nBaseline Model:")
    print(f"  Mean Accuracy: {baseline_mean:.4f}")
    print(f"  95% CI: [{baseline_lower:.4f}, {baseline_upper:.4f}]")
    print(f"  Std Dev: {np.std(baseline_cv_scores):.4f}")
    
    # 2. Test set results
    print("\n2. TEST SET RESULTS")
    print("-" * 60)
    y_pred = model.predict(X_test)
    y_pred_baseline = baseline.predict(X_test)
    
    test_acc = accuracy_score(y_test, y_pred)
    baseline_test_acc = accuracy_score(y_test, y_pred_baseline)
    
    print(f"Proposed Model Test Accuracy: {test_acc:.4f}")
    print(f"Baseline Model Test Accuracy: {baseline_test_acc:.4f}")
    print(f"Improvement: {(test_acc - baseline_test_acc) * 100:.2f}%")
    
    # Detailed metrics
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 3. Statistical significance
    print("\n3. STATISTICAL SIGNIFICANCE TEST")
    print("-" * 60)
    t_stat, p_value = ttest_rel(model_cv_scores, baseline_cv_scores)
    
    print(f"Paired t-test (CV scores):")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"  ✓ Statistically significant at α = 0.05")
    else:
        print(f"  ✗ Not statistically significant at α = 0.05")
    
    # 4. Effect size
    print("\n4. EFFECT SIZE ANALYSIS")
    print("-" * 60)
    d = cohens_d(model_cv_scores, baseline_cv_scores)
    
    print(f"Cohen's d: {d:.4f}")
    if abs(d) < 0.2:
        effect = "Negligible"
    elif abs(d) < 0.5:
        effect = "Small"
    elif abs(d) < 0.8:
        effect = "Medium"
    else:
        effect = "Large"
    print(f"Effect size: {effect}")
    
    # 5. Summary
    print("\n5. SUMMARY")
    print("=" * 60)
    
    stat_sig = p_value < 0.05
    practical_sig = abs(d) >= 0.5
    
    print(f"Statistical Significance: {'Yes' if stat_sig else 'No'} (p = {p_value:.4f})")
    print(f"Practical Significance: {'Yes' if practical_sig else 'No'} (d = {d:.4f})")
    print(f"Improvement over Baseline: {(test_acc - baseline_test_acc) * 100:.2f}%")
    
    if stat_sig and practical_sig:
        print("\n✓✓ CONCLUSION: Strong evidence of model improvement")
    elif stat_sig:
        print("\n✓✗ CONCLUSION: Statistically significant but small practical effect")
    elif practical_sig:
        print("\n✗✓ CONCLUSION: Large effect but not statistically significant (may need more data)")
    else:
        print("\n✗✗ CONCLUSION: No strong evidence of improvement")
    
    # Save results
    results = {
        'Metric': ['Model CV Accuracy', 'Baseline CV Accuracy', 'Model Test Accuracy', 
                   'Baseline Test Accuracy', 'P-value', 'Cohen\'s d'],
        'Value': [f"{model_mean:.4f}", f"{baseline_mean:.4f}", f"{test_acc:.4f}",
                  f"{baseline_test_acc:.4f}", f"{p_value:.4f}", f"{d:.4f}"],
        '95% CI': [f"[{model_lower:.4f}, {model_upper:.4f}]",
                   f"[{baseline_lower:.4f}, {baseline_upper:.4f}]",
                   'N/A', 'N/A', 'N/A', 'N/A']
    }
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('fyp_results_report.csv', index=False)
    print("\n📊 Results saved to 'fyp_results_report.csv'")
    
    return results_df

# Usage
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)

results = generate_fyp_results_report(model, baseline, X_train, y_train, X_test, y_test)
```

## FYP Submission Checklist

Before submitting your FYP, ensure:

- [ ] 95% confidence intervals reported for all key metrics
- [ ] Statistical significance tested (p-values reported)
- [ ] Effect sizes calculated (Cohen's d or Hedges' g)
- [ ] Comparison with appropriate baseline included
- [ ] Cross-validation results reported (not just single train/test split)
- [ ] Test set used only for final evaluation (not for model selection)
- [ ] Multiple metrics reported (not just accuracy)
- [ ] Results interpreted correctly (statistical vs practical significance)
- [ ] All random seeds documented for reproducibility
- [ ] Figures properly labeled with confidence intervals where applicable

## Common Mistakes to Avoid

1. **Testing on training data**: Always use separate test set for final evaluation
2. **P-hacking**: Don't try multiple tests until finding significance
3. **Ignoring practical significance**: Low p-value doesn't always mean meaningful improvement
4. **Overfitting to validation set**: Use proper nested CV for hyperparameter tuning
5. **Not using baseline**: Always compare with at least a simple baseline
6. **Cherry-picking metrics**: Report all relevant metrics, not just the best one
7. **Confusing CV and test performance**: They measure different things
8. **No confidence intervals**: Always report uncertainty in estimates
