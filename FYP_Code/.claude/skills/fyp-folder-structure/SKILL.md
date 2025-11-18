# FYP Folder Structure Enforcer

**Type:** Project Skill
**Purpose:** Enforce phase-based folder organization for TAR UMT FYP project

## When to Use This Skill

Use this skill when you need to:
- Create new files or folders
- Move, rename, or delete files
- Update file paths in code
- Check if a file location is correct
- Understand where outputs should be saved

**CRITICAL:** Always consult this skill BEFORE creating, moving, or deleting ANY file in the project.

---

## Phase-Based Folder Structure

```
FYP_Code/
├── notebooks/                    # All notebooks 00-16 (sequential)
│   ├── 00_environment_setup.ipynb
│   ├── 01_data_loading.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── ... (up to 16_flask_demo.ipynb)
│
├── experiments/                  # Phase-specific outputs
│   ├── phase1_exploration/       # Weeks 1-2
│   │   ├── eda_figures/          # EDA visualizations
│   │   ├── baseline_results/     # Initial baseline test results
│   │   └── augmentation_tests/   # Data augmentation experiments
│   │
│   ├── phase2_systematic/        # Weeks 3-6 (30 training runs)
│   │   ├── models/               # Trained model checkpoints
│   │   │   ├── resnet50/
│   │   │   ├── densenet121/
│   │   │   ├── efficientnet/
│   │   │   ├── vit/
│   │   │   ├── swin/
│   │   │   └── crossvit/
│   │   ├── results/
│   │   │   ├── confusion_matrices/
│   │   │   ├── metrics/
│   │   │   └── training_logs/
│   │   └── mlruns/               # MLflow experiment tracking
│   │
│   ├── phase3_analysis/          # Weeks 7-8
│   │   ├── statistical_validation/
│   │   ├── error_analysis/
│   │   └── ablation_studies/
│   │
│   └── phase4_deliverables/      # Weeks 9-10
│       ├── thesis_content/
│       │   ├── chapter4_tables/
│       │   └── chapter5_figures/
│       └── flask_demo/
│
├── data/                         # Dataset (IMMUTABLE)
│   ├── raw/                      # Original COVID-19 dataset (NEVER MODIFY)
│   └── processed/                # CSV files with train/val/test splits
│
├── src/                          # Reusable Python modules
│   ├── data_processing.py
│   ├── features.py
│   └── models.py
│
├── docs/                         # Documentation files
│   ├── *.md files
│   └── archive/                  # Old notebooks/docs
│
├── scripts/                      # Utility scripts
│   ├── train_*.py
│   ├── optimize_*.py
│   └── autonomous_*.py
│
├── logs/                         # Training logs
│   └── *.log files
│
├── config/                       # Configuration files (if needed)
│
└── references/                   # Papers, citations
```

---

## File Location Rules

### Rule 1: Notebooks Stay Sequential (00-16)

**✅ CORRECT:**
```
notebooks/00_environment_setup.ipynb
notebooks/01_data_loading.ipynb
notebooks/07_resnet50_training.ipynb
```

**❌ WRONG:**
```
notebooks/phase1/00_environment_setup.ipynb  # DON'T nest notebooks by phase
notebooks/models/07_resnet50_training.ipynb  # DON'T organize by topic
```

**Why:** Sequential numbering makes it easy to follow the workflow progression.

---

### Rule 2: Phase 1 Outputs → experiments/phase1_exploration/

**Scope:** Weeks 1-2 (Data loading, EDA, baseline testing)

**✅ CORRECT locations:**
```python
# EDA figures (from notebooks 03)
"experiments/phase1_exploration/eda_figures/class_distribution.png"
"experiments/phase1_exploration/eda_figures/sample_images.png"

# Baseline test results (from notebook 04)
"experiments/phase1_exploration/baseline_results/resnet50_baseline.png"
"experiments/phase1_exploration/baseline_results/baseline_metrics.csv"

# Augmentation experiments (from notebook 05)
"experiments/phase1_exploration/augmentation_tests/augmentation_comparison.png"
```

**❌ WRONG:**
```python
"results/eda_figures/class_distribution.png"  # Old structure
"notebooks/figures/sample_images.png"         # Don't save in notebooks/
"phase1_exploration/eda_figures/*.png"        # Missing experiments/ prefix
```

---

### Rule 3: Phase 2 Outputs → experiments/phase2_systematic/

**Scope:** Weeks 3-6 (Training 6 models × 5 seeds = 30 runs)

**✅ CORRECT locations:**

**Model Checkpoints:**
```python
"experiments/phase2_systematic/models/resnet50/resnet50_best_seed42.pth"
"experiments/phase2_systematic/models/crossvit/crossvit_best_seed123.pth"
```

**Confusion Matrices:**
```python
"experiments/phase2_systematic/results/confusion_matrices/resnet50_cm_seed42.png"
"experiments/phase2_systematic/results/confusion_matrices/crossvit_cm_seed789.png"
```

**Metrics CSVs:**
```python
"experiments/phase2_systematic/results/metrics/resnet50_results.csv"
"experiments/phase2_systematic/results/metrics/crossvit_results.csv"
```

**Training Logs:**
```python
"experiments/phase2_systematic/results/training_logs/resnet50_seed42.log"
```

**MLflow Tracking:**
```python
mlflow.set_tracking_uri("file:./experiments/phase2_systematic/mlruns")
```

**❌ WRONG:**
```python
"models/resnet50_best_seed42.pth"                    # Old structure
"results/confusion_matrices/resnet50_cm.png"         # Old structure
"experiments/phase2_systematic/resnet50_best.pth"    # Missing models/ subdirectory
```

---

### Rule 4: Phase 3 Outputs → experiments/phase3_analysis/

**Scope:** Weeks 7-8 (Statistical validation, error analysis)

**✅ CORRECT locations:**
```python
# Statistical validation (from notebook 12)
"experiments/phase3_analysis/statistical_validation/confidence_intervals.csv"
"experiments/phase3_analysis/statistical_validation/hypothesis_tests.csv"

# Error analysis (from notebook 13)
"experiments/phase3_analysis/error_analysis/misclassification_patterns.png"

# Ablation studies (from notebook 14)
"experiments/phase3_analysis/ablation_studies/clahe_comparison.png"
```

---

### Rule 5: Phase 4 Outputs → experiments/phase4_deliverables/

**Scope:** Weeks 9-10 (Thesis writing, Flask demo)

**✅ CORRECT locations:**
```python
# Thesis tables (from notebook 15)
"experiments/phase4_deliverables/thesis_content/chapter4_tables/reproducibility_statement.csv"
"experiments/phase4_deliverables/thesis_content/chapter5_figures/results_comparison.png"

# Flask demo (from notebook 16)
"experiments/phase4_deliverables/flask_demo/app.py"
"experiments/phase4_deliverables/flask_demo/templates/index.html"
```

---

### Rule 6: Data Files (IMMUTABLE)

**✅ CORRECT:**
```python
# Read-only access
"data/raw/COVID-19_Radiography_Dataset/COVID/images/*.png"  # NEVER MODIFY
"data/processed/train.csv"                                   # NEVER MODIFY
```

**❌ WRONG:**
```python
# NEVER write to data/raw/
with open("data/raw/new_file.csv", "w") as f:  # FORBIDDEN
    ...

# NEVER modify processed CSVs
df.to_csv("data/processed/train.csv")  # FORBIDDEN (already created)
```

**Why:** Data integrity. Once created, CSVs are reference datasets.

---

### Rule 7: Source Code → src/

**✅ CORRECT:**
```python
# Reusable modules
"src/data_processing.py"
"src/features.py"
"src/models.py"
```

**❌ WRONG:**
```python
"notebooks/helper_functions.py"  # Move to src/
"experiments/utils.py"            # Move to src/
```

---

### Rule 8: Scripts → scripts/

**✅ CORRECT:**
```python
# Standalone executable scripts
"scripts/train_all_models.py"
"scripts/autonomous_workflow.py"
"scripts/gpu_monitor.py"
```

**❌ WRONG:**
```python
"train_all_models.py"             # Move to scripts/
"notebooks/train_resnet50.py"     # Move to scripts/
```

---

### Rule 9: Documentation → docs/

**✅ CORRECT:**
```
docs/PHASE2_SETUP.md
docs/GPU_OPTIMIZATION_REPORT.md
docs/archive/04_baseline_test_FULL.ipynb
```

**❌ WRONG:**
```
PHASE2_SETUP.md                  # Move to docs/
notebooks/old_baseline.ipynb     # Move to docs/archive/
```

---

### Rule 10: Logs → logs/

**✅ CORRECT:**
```
logs/resnet50_training.log
logs/training_resnet50_ultra.log
```

**❌ WRONG:**
```
resnet50_training.log            # Move to logs/
experiments/training.log         # Move to logs/
```

---

## Code Examples

### Creating Files in Correct Locations

**Phase 1 - EDA Figure:**
```python
import matplotlib.pyplot as plt
from pathlib import Path

# CORRECT
output_dir = Path("experiments/phase1_exploration/eda_figures")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "class_distribution.png", dpi=300)
```

**Phase 2 - Model Checkpoint:**
```python
import torch
from pathlib import Path

# CORRECT
model_dir = Path("experiments/phase2_systematic/models/resnet50")
model_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), model_dir / "resnet50_best_seed42.pth")
```

**Phase 2 - Confusion Matrix:**
```python
import seaborn as sns
from pathlib import Path

# CORRECT
cm_dir = Path("experiments/phase2_systematic/results/confusion_matrices")
cm_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(cm_dir / "resnet50_cm_seed42.png", dpi=300)
```

**Phase 2 - MLflow Tracking:**
```python
import mlflow

# CORRECT
mlflow.set_tracking_uri("file:./experiments/phase2_systematic/mlruns")
mlflow.set_experiment("crossvit-covid19-classification")

with mlflow.start_run(run_name="resnet50-seed-42"):
    mlflow.log_param("model", "ResNet-50")
    mlflow.log_metric("test_accuracy", 94.5)
```

---

## Checking Rules Before Actions

### Before Creating a File:

**Ask:**
1. What phase does this output belong to?
2. What type of output is it? (model, result, figure, metric, log)
3. What is the correct subdirectory?

**Example Decision Tree:**
```
Creating a confusion matrix from ResNet-50 training?
├─ Phase? → Phase 2 (systematic training)
├─ Type?  → Confusion matrix (figure/result)
└─ Path?  → experiments/phase2_systematic/results/confusion_matrices/
```

### Before Moving a File:

**Ask:**
1. Does this file belong in the current location?
2. If not, which phase does it belong to?
3. Does the target directory exist?

### Before Deleting a File:

**Ask:**
1. Is this in data/raw/? → **NEVER DELETE**
2. Is this in data/processed/? → **VERIFY FIRST** (CSVs are references)
3. Is this a duplicate or obsolete output? → Safe to delete

---

## Common Mistakes to Avoid

### ❌ MISTAKE 1: Saving to Root Directory
```python
plt.savefig("confusion_matrix.png")  # WRONG: saves to root
```
**✅ FIX:**
```python
plt.savefig("experiments/phase2_systematic/results/confusion_matrices/resnet50_cm_seed42.png")
```

---

### ❌ MISTAKE 2: Using Old paths (models/, results/)
```python
torch.save(model, "models/resnet50_best.pth")  # WRONG: old structure
```
**✅ FIX:**
```python
torch.save(model, "experiments/phase2_systematic/models/resnet50/resnet50_best_seed42.pth")
```

---

### ❌ MISTAKE 3: Wrong MLflow Path
```python
mlflow.set_tracking_uri("file:./mlruns")  # WRONG: old location
```
**✅ FIX:**
```python
mlflow.set_tracking_uri("file:./experiments/phase2_systematic/mlruns")
```

---

### ❌ MISTAKE 4: Nested Notebooks
```python
# WRONG: Don't organize notebooks by phase
notebooks/phase2/07_resnet50_training.ipynb
```
**✅ FIX:**
```python
# Notebooks stay flat and sequential
notebooks/07_resnet50_training.ipynb
```

---

## Quick Reference Table

| Output Type | Phase | Correct Location |
|------------|-------|------------------|
| EDA figures | 1 | `experiments/phase1_exploration/eda_figures/` |
| Baseline results | 1 | `experiments/phase1_exploration/baseline_results/` |
| Augmentation tests | 1 | `experiments/phase1_exploration/augmentation_tests/` |
| Model checkpoints | 2 | `experiments/phase2_systematic/models/{model_name}/` |
| Confusion matrices | 2 | `experiments/phase2_systematic/results/confusion_matrices/` |
| Metrics CSVs | 2 | `experiments/phase2_systematic/results/metrics/` |
| Training logs | 2 | `experiments/phase2_systematic/results/training_logs/` |
| MLflow tracking | 2 | `experiments/phase2_systematic/mlruns/` |
| Statistical validation | 3 | `experiments/phase3_analysis/statistical_validation/` |
| Error analysis | 3 | `experiments/phase3_analysis/error_analysis/` |
| Ablation studies | 3 | `experiments/phase3_analysis/ablation_studies/` |
| Thesis tables | 4 | `experiments/phase4_deliverables/thesis_content/chapter4_tables/` |
| Thesis figures | 4 | `experiments/phase4_deliverables/thesis_content/chapter5_figures/` |
| Flask demo | 4 | `experiments/phase4_deliverables/flask_demo/` |

---

## Enforcement Checklist

Before ANY file operation, verify:

- [ ] Notebooks stay in `notebooks/` numbered 00-16
- [ ] Phase 1 outputs → `experiments/phase1_exploration/`
- [ ] Phase 2 outputs → `experiments/phase2_systematic/`
- [ ] Phase 3 outputs → `experiments/phase3_analysis/`
- [ ] Phase 4 outputs → `experiments/phase4_deliverables/`
- [ ] Data stays in `data/` (never modify raw/)
- [ ] Code stays in `src/` (reusable) or `scripts/` (executables)
- [ ] Docs stay in `docs/`
- [ ] Logs stay in `logs/`
- [ ] MLflow uses `experiments/phase2_systematic/mlruns`

---

## Summary

**Golden Rules:**
1. **Notebooks = Sequential (00-16)** in `notebooks/`
2. **Experiments = Phase-Based** in `experiments/phase{1-4}/`
3. **Data = Immutable** in `data/`
4. **Code = Modular** in `src/` or `scripts/`
5. **MLflow = Unified** in `experiments/phase2_systematic/mlruns`

**When in doubt:**
- Check this skill
- Ask: "What phase does this belong to?"
- Follow the phase-based structure

**NEVER:**
- Save to root directory
- Modify `data/raw/`
- Use old paths (`models/`, `results/`)
- Nest notebooks by phase
