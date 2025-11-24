"""
Run Statistical Validation (Phase 3)
Executes key analyses from 12_statistical_validation.ipynb
"""

import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_rel

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# Configuration
RESULTS_DIR = Path('experiments/phase2_systematic/results/metrics')
OUTPUT_DIR = Path('experiments/phase3_analysis/statistical_validation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

models = {
    'CrossViT-Tiny': 'crossvit_results.csv',
    'ResNet-50': 'resnet50_results.csv',
    'DenseNet-121': 'densenet121_results.csv',
    'EfficientNet-B0': 'efficientnet_results.csv',
    'ViT-Tiny': 'vit_results.csv',
    'Swin-Tiny': 'swin_results.csv'
}

ALPHA = 0.05
N_COMPARISONS = 5
BONFERRONI_ALPHA = ALPHA / N_COMPARISONS

print('='*80)
print('STATISTICAL VALIDATION - PHASE 3')
print('='*80)
print(f'\nSignificance level: alpha = {ALPHA}')
print(f'Bonferroni-corrected alpha\' = {BONFERRONI_ALPHA:.4f}')

# 1. Load all model results
print('\n1. LOADING RESULTS')
print('-'*80)
results = {}
for model_name, csv_file in models.items():
    df = pd.read_csv(RESULTS_DIR / csv_file)
    results[model_name] = df['test_acc'].values
    print(f'{model_name:20s}: {df["test_acc"].values}')

print('\n[OK] All results loaded')

# 2. Calculate 95% Confidence Intervals (Bootstrap)
print('\n2. CALCULATING 95% CONFIDENCE INTERVALS (Bootstrap, n=10000)')
print('-'*80)

def bootstrap_ci(data, n_bootstrap=10000, confidence=0.95):
    means = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))

    means = np.array(means)
    alpha = 1 - confidence
    lower = np.percentile(means, alpha/2 * 100)
    upper = np.percentile(means, (1 - alpha/2) * 100)

    return np.mean(data), lower, upper

ci_results = {}
print(f"{'Model':<20s} {'Mean':<12s} {'95% CI':<25s}")
print("-"*70)

for model_name, accuracies in results.items():
    mean, lower, upper = bootstrap_ci(accuracies)
    ci_results[model_name] = {'mean': mean, 'lower': lower, 'upper': upper}
    print(f"{model_name:<20s} {mean:>6.2f}%     [{lower:>6.2f}%, {upper:>6.2f}%]")

print("\n[OK] Confidence intervals calculated")

# 3. Hypothesis Testing
print('\n3. HYPOTHESIS TESTING: CrossViT-Tiny vs Baselines')
print('='*90)

def cohens_d(group1, group2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_std

crossvit_acc = results['CrossViT-Tiny']

print(f"{'Comparison':<30s} {'Mean Diff':<12s} {'t-stat':<10s} {'p-value':<12s} {'Significant':<15s} {'Cohen''s d'}")
print("-"*90)

hypothesis_results = []
baseline_models = ['ResNet-50', 'DenseNet-121', 'EfficientNet-B0', 'ViT-Tiny', 'Swin-Tiny']

for baseline in baseline_models:
    baseline_acc = results[baseline]
    t_stat, p_value = ttest_rel(crossvit_acc, baseline_acc)
    effect_size = cohens_d(crossvit_acc, baseline_acc)
    significant = "Yes*" if p_value < BONFERRONI_ALPHA else "No"
    mean_diff = np.mean(crossvit_acc) - np.mean(baseline_acc)

    print(f"CrossViT vs {baseline:<15s} {mean_diff:>+6.2f}%     {t_stat:>+6.3f}    {p_value:>8.4f}    {significant:<15s} {effect_size:>+6.3f}")

    hypothesis_results.append({
        'Comparison': f'CrossViT vs {baseline}',
        'Mean Difference (%)': mean_diff,
        't-statistic': t_stat,
        'p-value': p_value,
        "Significant (a' = 0.01)": significant,
        "Cohen's d": effect_size
    })

print("-"*90)
print("* Significant at Bonferroni-corrected alpha' = 0.01")
print("\n[OK] Hypothesis testing complete")

# Save results
hypothesis_df = pd.DataFrame(hypothesis_results)
hypothesis_df.to_csv(OUTPUT_DIR / 'hypothesis_testing_results.csv', index=False)
print(f"[OK] Results saved to: {OUTPUT_DIR / 'hypothesis_testing_results.csv'}")

# 4. Visualization
print('\n4. GENERATING VISUALIZATION')
print('-'*80)

fig, ax = plt.subplots(figsize=(12, 8))

model_names = list(ci_results.keys())
means = [ci_results[m]['mean'] for m in model_names]
lowers = [ci_results[m]['lower'] for m in model_names]
uppers = [ci_results[m]['upper'] for m in model_names]

# Sort by mean accuracy
sorted_indices = np.argsort(means)[::-1]
model_names = [model_names[i] for i in sorted_indices]
means = [means[i] for i in sorted_indices]
lowers = [lowers[i] for i in sorted_indices]
uppers = [uppers[i] for i in sorted_indices]

y_pos = np.arange(len(model_names))

# Plot CIs
colors = ['#FF6B6B' if 'CrossViT' in m else '#4ECDC4' for m in model_names]
ax.barh(y_pos, means, xerr=[[m-l for m, l in zip(means, lowers)],
                             [u-m for m, u in zip(means, uppers)]],
        color=colors, alpha=0.7, capsize=5)

ax.set_yticks(y_pos)
ax.set_yticklabels(model_names)
ax.set_xlabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Model Performance with 95% Confidence Intervals (Bootstrap, n=10000)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confidence_intervals_plot.png', dpi=300, bbox_inches='tight')
print(f"[OK] CI plot saved to: {OUTPUT_DIR / 'confidence_intervals_plot.png'}")
plt.close()

# 5. Generate Summary Report
print('\n5. GENERATING SUMMARY REPORT')
print('-'*80)

with open(OUTPUT_DIR / 'statistical_validation_summary.txt', 'w') as f:
    f.write("STATISTICAL VALIDATION SUMMARY\n")
    f.write("="*80 + "\n\n")

    f.write("1. DESCRIPTIVE STATISTICS\n")
    f.write("-"*80 + "\n")
    for model_name in model_names:
        mean = ci_results[model_name]['mean']
        sd = np.std(results[model_name], ddof=1)
        lower = ci_results[model_name]['lower']
        upper = ci_results[model_name]['upper']
        f.write(f"{model_name}: M = {mean:.2f}%, SD = {sd:.2f}, 95% CI [{lower:.2f}, {upper:.2f}]\n")

    f.write("\n2. HYPOTHESIS TESTING (Bonferroni-corrected a' = 0.01)\n")
    f.write("-"*80 + "\n")
    for result in hypothesis_results:
        comp = result['Comparison']
        diff = result['Mean Difference (%)']
        t = result['t-statistic']
        p = result['p-value']
        d = result["Cohen's d"]
        sig = result["Significant (a' = 0.01)"]
        f.write(f"{comp}: Diff = {diff:+.2f}%, t = {t:+.3f}, p = {p:.4f}, d = {d:+.3f}, {sig}\n")

    f.write("\n3. CONCLUSION\n")
    f.write("-"*80 + "\n")
    f.write("H1: CrossViT achieves significantly higher accuracy than CNN baselines\n")

    # Determine conclusion
    crossvit_better = sum(1 for r in hypothesis_results if r['Mean Difference (%)'] > 0 and r["Significant (a' = 0.01)"] == 'Yes*')
    crossvit_worse = sum(1 for r in hypothesis_results if r['Mean Difference (%)'] < 0 and r["Significant (a' = 0.01)"] == 'Yes*')

    if crossvit_better > 0:
        f.write(f"Result: PARTIALLY SUPPORTED - CrossViT significantly better than {crossvit_better} model(s)\n")
    elif crossvit_worse > 0:
        f.write(f"Result: NOT SUPPORTED - CrossViT significantly worse than {crossvit_worse} model(s)\n")
    else:
        f.write("Result: NO SIGNIFICANT DIFFERENCES - All models perform similarly\n")

print(f"[OK] Summary report saved to: {OUTPUT_DIR / 'statistical_validation_summary.txt'}")

# 6. APA-Formatted Table
print('\n6. APA-FORMATTED TABLE (FOR THESIS CHAPTER 5)')
print('='*90)
print()
print("Table 1")
print("Descriptive Statistics and 95% Confidence Intervals for Model Performance")
print()
print(f"{'Model':<20s} {'M':<8s} {'SD':<8s} {'95% CI':<25s} {'N'}")
print("-"*70)

for model_name in model_names:
    mean = ci_results[model_name]['mean']
    sd = np.std(results[model_name], ddof=1)
    lower = ci_results[model_name]['lower']
    upper = ci_results[model_name]['upper']
    n = len(results[model_name])

    print(f"{model_name:<20s} {mean:>5.2f}   {sd:>5.2f}   [{lower:>5.2f}, {upper:>5.2f}]       {n}")

print()
print("Note. M = mean accuracy (%), SD = standard deviation, CI = confidence interval, N = number of random seeds.")
print("Confidence intervals calculated using bootstrap method with 10,000 iterations.")

print('\n' + '='*80)
print('STATISTICAL VALIDATION COMPLETE')
print('='*80)
print('\nGenerated files:')
print('1. hypothesis_testing_results.csv - Detailed hypothesis tests')
print('2. confidence_intervals_plot.png - CI visualization')
print('3. statistical_validation_summary.txt - Text summary')
print('\nNext: Run 13_error_analysis.ipynb')
