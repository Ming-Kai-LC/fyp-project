# Experiments Folder - Phase-Based Organization

This folder organizes all experimental outputs by FYP phase following the CRISP-DM methodology.

## Structure

```
experiments/
├── phase1_exploration/          # Weeks 1-2: Dataset understanding
│   ├── eda_figures/             # 5 EDA visualizations
│   ├── baseline_results/        # Initial baseline test results
│   └── augmentation_tests/      # Data augmentation experiments
│
├── phase2_systematic/           # Weeks 3-6: Model training (30 runs)
│   ├── models/                  # Trained model checkpoints
│   │   ├── resnet50/
│   │   ├── densenet121/
│   │   ├── efficientnet/
│   │   ├── vit/
│   │   ├── swin/
│   │   └── crossvit/
│   ├── results/
│   │   ├── confusion_matrices/  # Per-seed confusion matrices
│   │   ├── metrics/             # CSV files with accuracy/loss
│   │   └── training_logs/       # Detailed training outputs
│   └── mlruns/                  # MLflow experiment tracking
│
├── phase3_analysis/             # Weeks 7-8: Statistical validation
│   ├── statistical_validation/  # 95% CIs, hypothesis tests
│   ├── error_analysis/          # Misclassification analysis
│   └── ablation_studies/        # H2, H3, H4 hypothesis testing
│
└── phase4_deliverables/         # Weeks 9-10: Thesis & deployment
    ├── thesis_content/
    │   ├── chapter4_tables/     # Reproducibility tables
    │   └── chapter5_figures/    # Publication-ready results
    └── flask_demo/              # Web interface prototype
```

## Current Status

- **Phase 1:** ✅ Complete (5 EDA figures saved)
- **Phase 2:** 🔄 In Progress (ResNet-50 training started)
- **Phase 3:** ⏸️ Not started
- **Phase 4:** ⏸️ Not started

## Usage

### Phase 2 Training (Current)

All training scripts now save to phase-specific locations:

```python
# Models saved to:
experiments/phase2_systematic/models/{model_name}/{model_name}_best_seed{seed}.pth

# Results saved to:
experiments/phase2_systematic/results/confusion_matrices/{model_name}_cm_seed{seed}.png
experiments/phase2_systematic/results/metrics/{model_name}_results.csv

# MLflow tracking:
experiments/phase2_systematic/mlruns/
```

### Viewing MLflow Results

```bash
# From project root:
mlflow ui --backend-store-uri file:./experiments/phase2_systematic/mlruns
# Open http://localhost:5000
```

### Expected Outputs

**Phase 2 (30 experiments):**
- 30 model checkpoints (.pth files)
- 30 confusion matrices (.png files)
- 6 results CSVs (one per model)
- 30 MLflow runs (tracked automatically)

**Phase 3:**
- Statistical validation tables (95% CIs)
- Hypothesis test results (H1-H4)
- Error analysis visualizations

**Phase 4:**
- All tables for thesis Chapter 5
- All figures for thesis Chapter 5
- Flask demo files

## Notes

- **Notebooks remain in `/notebooks/`** numbered 00-16 sequentially
- **Data stays in `/data/`** (immutable, shared across phases)
- **MLflow is unified** in phase2_systematic for easy comparison
- Each phase builds on previous outputs
