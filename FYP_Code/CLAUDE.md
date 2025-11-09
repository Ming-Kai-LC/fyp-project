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

## Project Architecture

### Data Flow Pipeline

The project follows a strict CRISP-DM methodology with 6 sequential phases:

1. **Data Loading** (`notebooks/01_data_loading.ipynb`) → Load COVID-19 Radiography Dataset (21,165 images, 4 classes)
2. **Data Cleaning** (`notebooks/02_data_cleaning.ipynb`) → CLAHE enhancement (clip=2.0, tile=8×8), validation checks
3. **EDA** (`notebooks/03_eda.ipynb`) → Class distribution analysis, image statistics, correlation studies
4. **Feature Engineering** (`notebooks/04_feature_engineering.ipynb`) → Preprocessing pipeline, normalization (ImageNet stats)
5. **Modeling** (`notebooks/05_modeling.ipynb`) → Train CrossViT + 5 baselines (ResNet-50, DenseNet-121, EfficientNet-B0, ViT-B/16, Swin-Tiny)
6. **Validation** (`notebooks/06_validation.ipynb`) → Statistical tests (95% CI, paired t-test, McNemar, DeLong), hypothesis testing

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

Test hypotheses in `notebooks/06_validation.ipynb` and `notebooks/07_ablation_studies.ipynb`.

## Skills Available

Use Claude Code skills for specialized tasks:

- **`/crossvit-covid19-fyp`**: Complete FYP context (dataset specs, model architecture, academic requirements)
- **`/fyp-jupyter`**: Data science workflow guide (CRISP-DM, preprocessing methods, statistical validation)
- **`/jupyter`**: General Jupyter notebook assistance

Invoke with: `@crossvit-covid19-fyp` or use the Skill tool.

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
- Test on small subset before full training
- Save intermediate outputs (preprocessed data, trained models)
- Monitor GPU memory during training
- Include proper error handling (CUDA OOM, file not found)
- Use `tqdm` progress bars for long operations

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
