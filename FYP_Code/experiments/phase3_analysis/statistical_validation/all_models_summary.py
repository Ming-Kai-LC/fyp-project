"""
Generate Summary Table for All 30 Training Runs
Author: Tan Ming Kai (24PMR12003)
Date: 2025-11-24
Purpose: Consolidate all Phase 2 results for thesis Chapter 5
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
RESULTS_DIR = Path("../../phase2_systematic/results/metrics")

# Load all results
models = {
    'CrossViT-Tiny': 'crossvit_results.csv',
    'ResNet-50': 'resnet50_results.csv',
    'DenseNet-121': 'densenet121_results.csv',
    'EfficientNet-B0': 'efficientnet_results.csv',
    'ViT-Tiny': 'vit_results.csv',
    'Swin-Tiny': 'swin_results.csv'
}

print("="*80)
print("PHASE 2 RESULTS SUMMARY: ALL 30 TRAINING RUNS")
print("="*80)
print()

# Detailed results per model
all_data = []
for model_name, csv_file in models.items():
    df = pd.read_csv(RESULTS_DIR / csv_file)

    print(f"{model_name}:")
    print(f"  {'Seed':<10} {'Test Accuracy':<15} {'Test Loss':<12} {'Training Time (min)':<20}")
    print(f"  {'-'*60}")

    for _, row in df.iterrows():
        acc = row['test_acc']
        loss = row['test_loss']
        time_min = row['training_time'] / 60 if row['training_time'] > 0 else 0
        seed = row['seed']

        print(f"  {seed:<10} {acc:>6.2f}%{'':<8} {loss:>6.4f}{'':<6} {time_min:>6.1f} min")

        all_data.append({
            'Model': model_name,
            'Seed': seed,
            'Test Accuracy (%)': acc,
            'Test Loss': loss,
            'Training Time (min)': time_min
        })

    # Statistics
    accuracies = df['test_acc'].values
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)

    print(f"  {'-'*60}")
    print(f"  Mean ± Std: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  Range: [{min_acc:.2f}%, {max_acc:.2f}%]")
    print()

# Save detailed CSV
detailed_df = pd.DataFrame(all_data)
detailed_df.to_csv('detailed_results_all_30_runs.csv', index=False)
print(f"[OK] Detailed results saved to: detailed_results_all_30_runs.csv")
print()

# Summary statistics table (for thesis)
print("="*80)
print("SUMMARY STATISTICS TABLE (FOR THESIS CHAPTER 5)")
print("="*80)
print()

summary_data = []
for model_name, csv_file in models.items():
    df = pd.read_csv(RESULTS_DIR / csv_file)
    accuracies = df['test_acc'].values

    summary_data.append({
        'Model': model_name,
        'N': len(accuracies),
        'Mean Accuracy (%)': np.mean(accuracies),
        'Std Dev (%)': np.std(accuracies, ddof=1),
        'Min (%)': np.min(accuracies),
        'Max (%)': np.max(accuracies),
        'Median (%)': np.median(accuracies)
    })

summary_df = pd.DataFrame(summary_data)

# Sort by mean accuracy (descending)
summary_df = summary_df.sort_values('Mean Accuracy (%)', ascending=False).reset_index(drop=True)

# Display formatted table
print(f"{'Model':<20} {'N':<5} {'Mean ± Std':<20} {'Min':<10} {'Max':<10} {'Median':<10}")
print(f"{'-'*85}")
for _, row in summary_df.iterrows():
    model = row['Model']
    n = int(row['N'])
    mean = row['Mean Accuracy (%)']
    std = row['Std Dev (%)']
    min_val = row['Min (%)']
    max_val = row['Max (%)']
    median = row['Median (%)']

    print(f"{model:<20} {n:<5} {mean:>5.2f} ± {std:<5.2f}{'%':<8} {min_val:>5.2f}%{'':<4} {max_val:>5.2f}%{'':<4} {median:>5.2f}%")

print()
summary_df.to_csv('summary_statistics_table.csv', index=False)
print(f"[OK] Summary statistics saved to: summary_statistics_table.csv")
print()

# LaTeX table (for thesis)
print("="*80)
print("LATEX TABLE (FOR THESIS)")
print("="*80)
print()
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Performance Comparison of COVID-19 Classification Models (5 Seeds)}")
print("\\label{tab:model_comparison}")
print("\\begin{tabular}{lcccc}")
print("\\hline")
print("\\textbf{Model} & \\textbf{Mean Accuracy (\\%)} & \\textbf{Std Dev (\\%)} & \\textbf{Min (\\%)} & \\textbf{Max (\\%)} \\\\")
print("\\hline")
for _, row in summary_df.iterrows():
    model = row['Model'].replace('_', '\\_')
    mean = row['Mean Accuracy (%)']
    std = row['Std Dev (%)']
    min_val = row['Min (%)']
    max_val = row['Max (%)']
    print(f"{model} & {mean:.2f} & {std:.2f} & {min_val:.2f} & {max_val:.2f} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\end{table}")
print()

# APA-formatted results (for thesis text)
print("="*80)
print("APA-FORMATTED RESULTS (FOR THESIS TEXT)")
print("="*80)
print()
for _, row in summary_df.iterrows():
    model = row['Model']
    mean = row['Mean Accuracy (%)']
    std = row['Std Dev (%)']
    n = int(row['N'])
    print(f"{model} achieved a mean accuracy of {mean:.2f}% (SD = {std:.2f}, N = {n}).")
print()

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
