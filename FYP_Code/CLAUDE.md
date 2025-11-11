# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a TAR UMT Data Science Final Year Project implementing CrossViT (Cross-Attention Vision Transformer) for COVID-19 chest X-ray classification. Student: Tan Ming Kai (24PMR12003), Academic Year: 2025/26.

**Core Philosophy:** COMPLETION OVER PERFECTION - Target is to pass (50%+), not publication quality. Working code takes priority over optimal code given semester timeline and consumer hardware constraints (RTX 4060 8GB VRAM).

## Development Environment Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Verify GPU setup
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Experiment Tracking with MLflow

**CRITICAL:** Use MLflow to track all experiments. You will run 30+ training runs (6 models × 5 seeds), and manual tracking is error-prone and time-consuming.

### Why MLflow?

**Without MLflow:**
- Scattered results across notebooks
- Forgotten hyperparameters
- Difficult to compare models
- Manual table creation for thesis
- 5-10 hours wasted on spreadsheets

**With MLflow:**
- Central experiment tracking
- Automatic parameter logging
- Easy model comparison with sortable UI
- Direct export for thesis tables
- Industry-standard tool (Netflix, Databricks)

**Time Investment:** 5 minutes setup → Save 5-10 hours during FYP

### Setup (One-Time, 5 Minutes)

```bash
# Install MLflow
pip install mlflow

# Start MLflow UI (optional, for viewing results)
mlflow ui  # Open http://localhost:5000
```

### Basic Usage Pattern

```python
import mlflow

# Set experiment name (do ONCE per project)
mlflow.set_experiment("crossvit-covid19-classification")

# Log a training run
seeds = [42, 123, 456, 789, 101112]

for seed in seeds:
    with mlflow.start_run(run_name=f"crossvit-seed-{seed}"):
        # Log parameters (hyperparameters, config)
        mlflow.log_param("model", "CrossViT-Tiny")
        mlflow.log_param("random_seed", seed)
        mlflow.log_param("learning_rate", 5e-5)
        mlflow.log_param("batch_size", 8)
        mlflow.log_param("epochs", 50)

        # Train your model
        model = train_model(seed=seed)

        # Log metrics (results)
        accuracy = evaluate(model, test_loader)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)

        # Log artifacts (plots, models)
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # Optional: log the model itself
        mlflow.pytorch.log_model(model, "model")
```

### Viewing Results

```bash
# In terminal, run:
mlflow ui

# Then open: http://localhost:5000
# You'll see all experiments with:
# - Sortable columns (accuracy, loss, etc.)
# - Parameter comparison
# - Artifact downloads (plots, models)
```

### Export for Thesis

```python
# After all experiments, export results for Chapter 5
import mlflow
import pandas as pd

experiment = mlflow.get_experiment_by_name("crossvit-covid19-classification")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# View as DataFrame
print(runs[['params.model', 'params.random_seed', 'metrics.test_accuracy']])

# Export to CSV for thesis tables
runs.to_csv('all_experiment_results.csv', index=False)

# Calculate statistics per model
models = runs['params.model'].unique()
for model in models:
    model_runs = runs[runs['params.model'] == model]
    accuracies = model_runs['metrics.test_accuracy'].tolist()

    mean = np.mean(accuracies)
    std = np.std(accuracies)
    print(f"{model}: {mean:.1%} ± {std:.1%}")
```

### Best Practices

**DO:**
✅ Log every training run (even failed ones)
✅ Use consistent naming: `{model}-seed-{number}`
✅ Log all hyperparameters before training
✅ Save confusion matrices as artifacts
✅ Tag runs: `mlflow.set_tag("phase", "exploration")`

**DON'T:**
❌ Skip logging (you'll forget results!)
❌ Log raw datasets (too large)
❌ Use different experiment names for same project
❌ Forget to log random seeds

**When to Use:**
- **Phase 1:** Start logging baseline test runs
- **Phase 2:** Log all 30 systematic experiments
- **Phase 3:** Use logged data for statistical validation
- **Phase 4:** Export tables directly from MLflow

## Project Architecture

### Notebook Development Sequence (Phase-Based Approach)

The project follows a 4-phase workflow aligned with CRISP-DM methodology. Each phase has specific goals and deliverables.

#### Phase 1: Exploration (Weeks 1-2) ✅ CURRENT PHASE

**Goal:** Understand dataset + Get ONE baseline model working

**Notebooks:**
1. **00_environment_setup.ipynb** ✅ COMPLETED
   - Verify GPU, CUDA, dependencies
   - Test CrossViT model loading
   - Validate dataset paths

2. **01_data_loading.ipynb** ✅ COMPLETED
   - Load COVID-19 Radiography Dataset (21,165 images, 4 classes)
   - Create stratified train/val/test splits (80/10/10)
   - Save image paths to CSV

3. **02_data_cleaning.ipynb** ✅ COMPLETED
   - CLAHE enhancement (clip=2.0, tile=8×8)
   - Resize to 240×240
   - Convert grayscale → RGB
   - Validation checks

4. **03_eda.ipynb** ✅ COMPLETED
   - Class distribution analysis
   - Pixel intensity statistics
   - Statistical tests (ANOVA)
   - Generate publication-ready figures

5. **04_baseline_test.ipynb** ⏭️ NEXT STEP
   - Create PyTorch Dataset and DataLoader
   - Train ONE baseline model (ResNet-50 recommended)
   - Verify training pipeline works end-to-end
   - Test on small subset first (1000 images)
   - **Goal:** Achieve >70% accuracy, confirm GPU memory safe

6. **05_augmentation_test.ipynb** ⏭️ AFTER BASELINE
   - Test augmentation strategies
   - Compare: No aug vs Conservative aug vs Aggressive aug
   - Finalize data augmentation pipeline for Phase 2
   - Implement ImageNet normalization

**Phase 1 Success Criteria:**
- [ ] All notebooks 00-03 complete ✅ DONE
- [ ] ONE baseline model trains successfully
- [ ] Training pipeline reproducible (seed=42)
- [ ] GPU memory usage within limits (<7GB)
- [ ] Augmentation strategy finalized

#### Phase 2: Systematic Experimentation (Weeks 3-6)

**Goal:** Train ALL 6 models with 5 seeds each (30 total runs)

**Notebooks:**
7. **06_crossvit_training.ipynb** - CrossViT-Tiny (5 seeds: 42, 123, 456, 789, 101112)
8. **07_resnet50_training.ipynb** - Baseline 1 (5 seeds)
9. **08_densenet121_training.ipynb** - Baseline 2 (5 seeds)
10. **09_efficientnet_training.ipynb** - Baseline 3 (5 seeds)
11. **10_vit_training.ipynb** - Baseline 4 (5 seeds)
12. **11_swin_training.ipynb** - Baseline 5 (5 seeds)

**Each notebook:**
- Loads configuration from MLflow
- Trains with all 5 random seeds
- Logs results to MLflow automatically
- Saves best model checkpoints
- Generates confusion matrices

**Phase 2 Success Criteria:**
- [ ] All 6 models implemented
- [ ] 30 total training runs completed (6 models × 5 seeds)
- [ ] All runs logged in MLflow
- [ ] Confusion matrices and metrics saved
- [ ] Training reproducible

#### Phase 3: Analysis & Refinement (Weeks 7-8)

**Goal:** Statistical validation and deep analysis

**Notebooks:**
13. **12_statistical_validation.ipynb**
    - Calculate 95% CIs for all models
    - Paired t-test (CrossViT vs each baseline)
    - Bonferroni correction for multiple comparisons
    - Generate results tables for thesis

14. **13_error_analysis.ipynb**
    - Analyze misclassifications
    - Per-class performance breakdown
    - Failure case visualization
    - Identify patterns in errors

15. **14_ablation_studies.ipynb**
    - Test H2, H3, H4 hypotheses
    - CLAHE vs No CLAHE comparison
    - Augmentation impact analysis
    - Dual-branch vs single-scale

**Phase 3 Success Criteria:**
- [ ] 95% CI calculated for all models
- [ ] Hypothesis tests completed
- [ ] Statistical significance confirmed
- [ ] All tables/figures for Chapter 5 ready

#### Phase 4: Documentation & Deployment (Weeks 9-10)

**Goal:** Thesis writing and Flask demo

**Notebooks:**
16. **15_thesis_content.ipynb**
    - Generate all tables for Chapter 5
    - Format figures with captions
    - Export reproducibility statement for Chapter 4

17. **16_flask_demo.ipynb**
    - Prepare model for deployment
    - Test inference pipeline
    - Create basic web interface prototype

**Phase 4 Success Criteria:**
- [ ] Chapter 4 complete with reproducibility statement
- [ ] Chapter 5 complete with all results
- [ ] Flask demo functional (basic)
- [ ] All code documented and tested

### Code Organization

- **`src/`**: Reusable modules imported by notebooks
  - `data_processing.py`: Loading, missing value handling, outlier removal, feature scaling
  - `features.py`: Interaction features, ratio features, polynomial features, binning
  - `models.py`: Training loops, evaluation metrics, confidence intervals, model comparison, hyperparameter tuning

- **`data/`**: Data storage with strict immutability
  - `raw/`: Original COVID-19 dataset - **NEVER MODIFY**
  - `processed/`: CLAHE-enhanced images, train/val/test splits
  - `external/`: Third-party augmentation sources

- **`models/`**: Trained model artifacts (`.pkl`, `.h5`, `.pth`)
- **`results/`**: Publication-ready outputs
  - `figures/`: Confusion matrices, ROC curves, training plots (300 DPI)
  - `tables/`: Metrics CSVs with 95% confidence intervals

### Notebook Imports Pattern

All notebooks MUST follow this structure:

```python
# 1. REPRODUCIBILITY (always first)
import random, numpy as np, torch
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 2. Import custom modules
import sys
sys.path.append('../src')
from data_processing import *
from features import *
from models import *

# 3. Hardware verification
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
```

## Critical Hardware Constraints

**RTX 4060 8GB VRAM Limits:**
- Maximum batch_size: 16 (use 8 for safety)
- Use gradient accumulation (4 steps) for effective batch size of 32
- Enable mixed precision training (`torch.cuda.amp.autocast()`)
- Monitor memory with `torch.cuda.memory_allocated()` every 50 batches
- Clear cache every 10 batches: `torch.cuda.empty_cache()`
- DataLoader: `num_workers=4, pin_memory=True, persistent_workers=True`

```python
# Memory-safe training configuration
CONFIG = {
    'batch_size': 8,
    'gradient_accumulation_steps': 4,  # Effective batch = 32
    'mixed_precision': True,
    'num_workers': 4,
}
```

## Model Specifications

**CrossViT-Tiny** (Primary Model):
```python
import timm
model = timm.create_model('crossvit_tiny_240', pretrained=True, num_classes=4)
# Input: 240×240×3 RGB
# Dual branches: 16×16 and 12×12 patches
# ~7M parameters (fits in 8GB VRAM)
```

**Training Hyperparameters** (Fixed):
```python
optimizer = AdamW(lr=5e-5, weight_decay=0.05)
scheduler = CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
criterion = nn.CrossEntropyLoss(weight=[1.47, 0.52, 0.88, 3.95])  # Class imbalance
max_epochs = 50
early_stopping_patience = 15
```

## Data Preprocessing Pipeline

**EXACT specifications** (do not modify):

```python
# 1. CLAHE enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(grayscale_image)

# 2. Resize to 240×240
resized = cv2.resize(enhanced, (240, 240))

# 3. Convert grayscale to RGB (CrossViT requires 3 channels)
rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

# 4. Normalize to ImageNet statistics
from torchvision import transforms
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# 5. Training augmentation (apply AFTER CLAHE)
train_transform = transforms.Compose([
    transforms.RandomRotation(10),           # ±10° only
    transforms.RandomHorizontalFlip(0.5),    # NO vertical flip
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    normalize
])
```

## Statistical Validation Requirements

All results MUST include:

1. **95% Confidence Intervals** (bootstrap with 1000 iterations)
2. **Paired t-test** (30 runs, α=0.05, Bonferroni correction α'=0.01)
3. **Effect sizes** (Cohen's d)
4. **Medical metrics**: Sensitivity, Specificity, PPV, NPV

```python
from src.models import calculate_confidence_interval, compare_models

# Example usage
mean, lower, upper = calculate_confidence_interval(cv_scores, confidence=0.95)
results = compare_models(crossvit_model, resnet_model, X_train, y_train, cv=10)
```

## Research Hypotheses Testing

**H₁ (Primary):** CrossViT achieves significantly higher accuracy than CNN baselines (p<0.05)
**H₂:** Dual-branch processing improves accuracy by ≥5% vs single-scale
**H₃:** CLAHE enhancement improves performance by ≥2% vs no CLAHE
**H₄:** Conservative augmentation improves generalization without degrading accuracy

Test hypotheses in `notebooks/12_statistical_validation.ipynb` and `notebooks/14_ablation_studies.ipynb`.

## Current Phase & Weekly Goals

### Identifying Your Current Phase

**Answer these questions to know where you are:**

1. Do you have a working baseline model?
   - **No** → Phase 1 (Exploration)
   - **Yes** → Continue

2. Have you tested all 6 models with 5 seeds each (30 runs)?
   - **No** → Phase 2 (Systematic Experimentation)
   - **Yes** → Continue

3. Have you completed statistical validation (CIs, hypothesis tests)?
   - **No** → Phase 3 (Analysis & Refinement)
   - **Yes** → Phase 4 (Documentation)

### You Are Currently In: Phase 1 (Exploration) ✅

**What to Focus On This Week:**
- [ ] Complete `04_baseline_test.ipynb`
- [ ] Get ResNet-50 training successfully
- [ ] Verify GPU memory usage (<7GB)
- [ ] Test on small subset (1000 images) first
- [ ] Achieve >70% accuracy (not 95%!)

**What NOT to Worry About:**
❌ Hyperparameter optimization (use defaults)
❌ Training all 6 models yet
❌ Perfect accuracy (working code > optimal results)
❌ Statistical validation (too early)

**When to Move to Phase 2:**
✅ You have ONE model that trains successfully
✅ Training pipeline is reproducible (seed=42)
✅ You understand the workflow
✅ Augmentation strategy is finalized

**Time Check:**
- **Expected:** Week 1-2 of FYP
- **Behind schedule?** Focus on getting baseline working, skip augmentation testing for now
- **Ahead of schedule?** Good! Test augmentation before Phase 2

### Weekly Goals by Phase

**Phase 1 (Weeks 1-2):**
- Week 1: Environment setup + data loading + EDA ✅ DONE
- Week 2: Baseline model working + augmentation tested ⏭️ CURRENT

**Phase 2 (Weeks 3-6):**
- Week 3: Train CrossViT (5 seeds)
- Week 4: Train baselines 1-2 (ResNet, DenseNet)
- Week 5: Train baselines 3-5 (EfficientNet, ViT, Swin)
- Week 6: Verify all 30 runs, fix any failures

**Phase 3 (Weeks 7-8):**
- Week 7: Statistical validation (CIs, hypothesis tests)
- Week 8: Error analysis, ablation studies

**Phase 4 (Weeks 9-10):**
- Week 9: Write Chapter 4-5
- Week 10: Flask demo, final checks, submission

## Skills Available

Use Claude Code skills for specialized tasks. Each skill has a specific purpose:

### Core Skills (Use Daily)

1. **`@fyp-jupyter`** - Your daily workflow guide
   - **Use when:** "What should I work on today?"
   - **Provides:** Phase identification, weekly goals, CRISP-DM workflow
   - **Examples:**
     - "What phase am I in?"
     - "How do I handle missing values?"
     - "How to set up MLflow?"

2. **`@crossvit-covid19-fyp`** - Technical specifications
   - **Use when:** Need model/dataset/hardware details
   - **Provides:** CrossViT architecture, GPU constraints, hyperparameters
   - **Examples:**
     - "What learning rate for CrossViT?"
     - "What's the input size?"
     - "How much VRAM needed?"

3. **`@fyp-statistical-validator`** - Statistical validation & thesis formatting
   - **Use when:** Need to validate results or format for thesis
   - **Provides:** 95% CIs, hypothesis tests, APA-formatted tables
   - **Examples:**
     - "Calculate confidence interval"
     - "Is my result statistically significant?"
     - "Generate table for Chapter 5"

### Support Skills (Use When Needed)

4. **`@tar-umt-fyp-rds`** - FYP administrative requirements
   - **Use when:** Need to know deadlines, deliverables, structure
   - **Examples:** "When is Project I due?", "What chapters needed?"

5. **`@tar-umt-academic-writing`** - APA citations and plagiarism
   - **Use when:** Need to cite sources or check Turnitin
   - **Examples:** "How to cite this paper?", "Is 18% similarity okay?"

### Quick Decision Guide

**I need to...**
- Know what to work on → `@fyp-jupyter`
- Understand CrossViT specs → `@crossvit-covid19-fyp`
- Calculate statistics → `@fyp-statistical-validator`
- Check FYP deadlines → `@tar-umt-fyp-rds`
- Format citations → `@tar-umt-academic-writing`

**Invoke skills with:** `@skill-name` in your message

## Common Pitfalls

**MUST AVOID:**
- Modifying raw data in `data/raw/` (keep original immutable)
- Training without early stopping (wastes GPU time)
- Using batch_size > 16 (causes OOM on 8GB VRAM)
- Fitting scalers on full dataset before train/test split (data leakage)
- Vertical flipping chest X-rays (anatomically incorrect)
- Skipping reproducibility seeds (results not reproducible)

**ALWAYS DO:**
- Set `seed=42` for all random operations (Python, NumPy, PyTorch, CUDA)
- **Use MLflow to log all training runs** (parameters, metrics, artifacts)
- Test on small subset before full training (1000 images first)
- Save intermediate outputs (preprocessed data, trained models)
- Monitor GPU memory during training
- Include proper error handling (CUDA OOM, file not found)
- Use `tqdm` progress bars for long operations
- Start with Phase 1 baseline before systematic experiments

## Academic Requirements

- **Reproducibility**: All experiments must use seed=42
- **Statistical Rigor**: 95% CI and p-values for all comparisons
- **Documentation**: Every decision justified in notebook markdown cells
- **Attribution**: Use APA 7th Edition citations for algorithms from papers
- **Turnitin**: Code similarity must be <20%
- **Timeline**: Must complete within semester (prioritize completion over perfection)

## Dataset Specifications

**COVID-19 Radiography Database** (Rahman et al., 2021):
- Total: 21,165 chest X-rays (299×299 PNG, grayscale)
- Classes: COVID-19 (3,616), Normal (10,192), Lung Opacity (6,012), Viral Pneumonia (1,345)
- Split: 80% train (16,932) / 10% val (2,116) / 10% test (2,117)
- Class imbalance: 7.6:1 ratio (use weighted loss)
- Located in: `data/raw/COVID-19_Radiography_Dataset/`

## Success Criteria

**Minimum viable FYP** (to pass):
- CrossViT achieves >85% accuracy (not 95%)
- All 5 baselines trained and compared
- Statistical tests completed (H₁ validated with p<0.05)
- All notebooks run without errors
- Basic Flask demo works (functionality over aesthetics)

Remember: **DONE > PERFECT**. This is about graduation, not publication.
