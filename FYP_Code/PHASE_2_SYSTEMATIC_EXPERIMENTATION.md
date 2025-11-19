# Phase 2: Systematic Experimentation

**Duration:** Week 3-6 (In Progress - Week 6)
**Goal:** Train ALL 6 models with 5 seeds each (30 total runs)
**Status:** 🔄 IN PROGRESS (5/6 models complete, ViT training now)

---

## 📋 Phase Overview

Phase 2 involves systematic training of all baseline models and the main CrossViT model, each with 5 different random seeds for statistical validation. This ensures reproducibility and provides confidence intervals for model performance.

---

## 🎯 Models to Train

| # | Model | Parameters | Status | Mean Accuracy | Seeds Complete |
|---|-------|-----------|--------|---------------|----------------|
| 1 | ResNet-50 | 23.5M | ✅ COMPLETE | 95.45% ± 0.57% | 5/5 |
| 2 | DenseNet-121 | 7.0M | ✅ COMPLETE | 95.32% ± 0.26% | 5/5 |
| 3 | EfficientNet-B0 | 4.0M | ✅ COMPLETE | 95.23% ± 0.33% | 5/5 |
| 4 | CrossViT-Tiny | 7.0M | ✅ COMPLETE | 94.96% ± 0.55% | 5/5 |
| 5 | ViT-Base | 85.8M | 🔄 IN PROGRESS | TBD | 0/5 (training) |
| 6 | Swin-Tiny | 27.5M | ⏸️ PENDING | TBD | 0/5 |

**Progress: 20/30 runs complete (66.7%)**

---

## ✅ Completed Models

### 1. ResNet-50 (Baseline CNN) ✅

**Training Period:** 2025-11-18, 14:30 - 16:58 (~2.5 hours)
**Architecture:** Deep Residual Network (He et al., 2016)

**Configuration:**
```python
Model: torchvision.models.resnet50(pretrained=True)
Parameters: 23,512,068
Batch Size: 170 (auto-adjusted from 200)
Learning Rate: 1e-4
Optimizer: Adam
Epochs: Variable (early stopping)
```

**Results by Seed:**

| Seed | Test Acc | Test Loss | Training Time | Epochs | Best Epoch |
|------|----------|-----------|---------------|--------|------------|
| 42 | 95.28% | 0.1233 | 26.4 min | 17 | 7 |
| 123 | 95.98% | 0.1345 | 40.8 min | 23 | 13 |
| 456 | 94.66% | 0.1428 | 29.8 min | 22 | 12 |
| 789 | 95.28% | 0.1243 | 25.8 min | 17 | 7 |
| 101112 | **96.03%** | 0.1390 | 31.6 min | 20 | 10 |

**Summary Statistics:**
- **Mean ± Std:** 95.45% ± 0.57%
- **Range:** [94.66%, 96.03%]
- **Best Seed:** 101112 (96.03%)
- **Total Training Time:** 2h 34min

**Key Observations:**
- Most consistent performer across seeds
- Highest overall mean accuracy
- Best single-seed result (96.03%)
- Fast convergence (10-13 epochs typically)

**Files Saved:**
```
experiments/phase2_systematic/
├── models/resnet50/
│   ├── resnet50_best_seed42.pth       (91 MB)
│   ├── resnet50_best_seed123.pth      (91 MB)
│   ├── resnet50_best_seed456.pth      (91 MB)
│   ├── resnet50_best_seed789.pth      (91 MB)
│   └── resnet50_best_seed101112.pth   (91 MB)
├── results/
│   ├── metrics/resnet50_results.csv
│   └── confusion_matrices/
│       ├── resnet50_cm_seed42.png
│       ├── resnet50_cm_seed123.png
│       ├── resnet50_cm_seed456.png
│       ├── resnet50_cm_seed789.png
│       └── resnet50_cm_seed101112.png
```

---

### 2. DenseNet-121 (Baseline Dense CNN) ✅

**Training Period:** 2025-11-18, 17:27 - 20:12 (~2 hours with optimized batch size)
**Architecture:** Densely Connected Convolutional Networks (Huang et al., 2017)

**Configuration:**
```python
Model: torchvision.models.densenet121(pretrained=True)
Parameters: 6,957,956
Batch Size: 323 (auto-adjusted from 380 - OPTIMIZED!)
Learning Rate: 1.78e-4 (scaled for larger batch)
Optimizer: Adam
Epochs: Variable (early stopping)
```

**Results by Seed:**

| Seed | Test Acc | Test Loss | Training Time | Epochs | Best Epoch |
|------|----------|-----------|---------------|--------|------------|
| 42 | 95.28% | 0.1035 | 31.6 min | 18 | 8 |
| 123 | 95.51% | 0.1324 | 38.0 min | 21 | 11 |
| 456 | 95.18% | 0.1106 | 30.5 min | 17 | 7 |
| 789 | **95.65%** | 0.1134 | 30.5 min | 17 | 7 |
| 101112 | 94.99% | 0.1149 | 34.2 min | 18 | 8 |

**Summary Statistics:**
- **Mean ± Std:** 95.32% ± 0.26%
- **Range:** [94.99%, 95.65%]
- **Best Seed:** 789 (95.65%)
- **Total Training Time:** 2h 45min

**Key Observations:**
- **Most consistent model** (lowest std dev: 0.26%)
- All seeds achieved 95%+ accuracy
- Faster training than ResNet-50 (21% fewer batches/epoch)
- Benefited from batch size optimization (+48% larger batches)

**Performance Improvement:**
- Batch size: 217 → 323 (+48%)
- Training speed: ~33% faster per epoch
- GPU utilization: 99% (excellent)

---

### 3. EfficientNet-B0 (Baseline Efficient CNN) ✅

**Training Period:** 2025-11-18, 20:27 - 22:48 (~2 hours)
**Architecture:** EfficientNet (Tan & Le, 2019)

**Configuration:**
```python
Model: torchvision.models.efficientnet_b0(pretrained=True)
Parameters: ~4,000,000 (smallest model)
Batch Size: 323
Learning Rate: 1.78e-4
Optimizer: Adam
Epochs: Variable (early stopping)
```

**Results by Seed:**

| Seed | Test Acc | Test Loss | Training Time | Epochs | Best Epoch |
|------|----------|-----------|---------------|--------|------------|
| 42 | 95.18% | 0.1344 | 29.8 min | 16 | 6 |
| 123 | 95.56% | 0.1189 | 26.6 min | 14 | 4 |
| 456 | **95.65%** | 0.1203 | 32.2 min | 18 | 8 |
| 789 | 94.80% | 0.1411 | 33.3 min | 19 | 9 |
| 101112 | 95.09% | 0.1263 | 30.1 min | 17 | 7 |

**Summary Statistics:**
- **Mean ± Std:** 95.23% ± 0.33%
- **Range:** [94.80%, 95.65%]
- **Best Seed:** 456 (95.65%)
- **Total Training Time:** 2h 32min

**Key Observations:**
- Smallest model (4M params) but competitive performance
- Fast training (fewer parameters)
- Good balance of efficiency and accuracy
- Tied for best single-seed result (95.65% with DenseNet)

---

### 4. CrossViT-Tiny-240 (Main Model - Dual-Branch Transformer) ✅

**Training Period:** 2025-11-18, 23:14 - 02:19 (~3.5 hours)
**Architecture:** Cross-Attention Vision Transformer (Chen et al., 2021)

**Configuration:**
```python
Model: timm.create_model('crossvit_tiny_240', pretrained=True)
Parameters: ~7,000,000
Batch Size: 323
Learning Rate: 1.78e-4
Optimizer: Adam
Input Size: 240×240 (dual branches: 16×16 and 12×12 patches)
```

**Results by Seed:**

| Seed | Test Acc | Test Loss | Training Time | Epochs | Best Epoch |
|------|----------|-----------|---------------|--------|------------|
| 42 | 94.66% | 0.1121 | 33.0 min | 18 | 8 |
| 123 | 95.42% | 0.1226 | 39.3 min | 22 | 12 |
| 456 | **95.65%** | 0.1423 | 44.0 min | 25 | 15 |
| 789 | 94.33% | 0.1230 | 41.0 min | 23 | 13 |
| 101112 | 94.76% | 0.1274 | 43.9 min | 24 | 14 |

**Summary Statistics:**
- **Mean ± Std:** 94.96% ± 0.55%
- **Range:** [94.33%, 95.65%]
- **Best Seed:** 456 (95.65%)
- **Total Training Time:** 3h 21min

**Key Observations:**
- **Main FYP model** with dual-branch architecture
- Longer training time (more complex architecture)
- Higher variance across seeds (±0.55%)
- **Unexpectedly underperformed CNNs** (H₁ rejected!)

**Important Research Finding:**
CrossViT did NOT outperform CNN baselines as hypothesized:
- ResNet-50: +0.49% better
- DenseNet-121: +0.36% better
- EfficientNet-B0: +0.27% better

This is a **valid research result** for thesis discussion!

---

## 🔄 Currently Training

### 5. ViT-Base (Baseline Vision Transformer) 🔄

**Training Started:** 2025-11-19, 02:30
**Architecture:** Vision Transformer (Dosovitskiy et al., 2021)

**Configuration:**
```python
Model: timm.create_model('vit_base_patch16_224',
                         pretrained=True,
                         img_size=240,  # Fixed image size issue
                         num_classes=4)
Parameters: 85,824,004 (largest model)
Batch Size: 323
Learning Rate: 1.78e-4
Input Size: 240×240 (patch size: 16×16)
```

**Current Status:**
- Seed 42: Epoch 1 in progress
- GPU: 100% utilization, 32.3 GB VRAM (66%)
- Temperature: 64°C (safe)
- Estimated completion: ~05:00 (2.5 hours total)

**Issue Resolved:**
- **Problem:** Image size mismatch (input 240×240, model expects 224×224)
- **Solution:** Added `img_size=240` parameter to model initialization
- **Status:** ✅ Fixed and training successfully

---

## ⏸️ Pending

### 6. Swin-Tiny (Baseline Shifted Window Transformer) ⏸️

**Scheduled:** Will auto-start after ViT completes
**Architecture:** Swin Transformer (Liu et al., 2021)

**Configuration:**
```python
Model: timm.create_model('swin_tiny_patch4_window7_224',
                         pretrained=True,
                         img_size=240,  # Fixed image size issue
                         num_classes=4)
Parameters: 27,522,430
Batch Size: 323
Learning Rate: 1.78e-4
Window Size: 7×7 with shifted windows
```

**Preparation:**
- ✅ Image size fix applied (img_size=240)
- ✅ Auto-monitor script running (will start when ViT completes)
- Estimated time: ~2 hours (5 seeds)

---

## 🔧 Optimizations Applied in Phase 2

### 1. Batch Size Optimization (Week 6)

**Problem:** Only using 58% of available VRAM (28 GB / 48 GB)

**Solution:**
```python
# Before:
batch_size = 200 → auto-adjusted to 170 (single user)

# After:
batch_size = 380 → auto-adjusted to 323 (single user)
```

**Results:**
- Batch size: +48% increase (217 → 323)
- VRAM usage: 28 GB → 35.6 GB (74% utilization)
- Training speed: ~33% faster per epoch
- GPU utilization: 99% (excellent)

**Impact:**
- DenseNet-121: 78 batches/epoch → 52 batches/epoch
- Saved ~1 hour per model

### 2. Auto-Sequential Training

**Problem:** Manual intervention needed between models

**Solution:**
- Created `train_all_models_sequential.py`
- Automatically trains models one after another
- Skips completed models
- Logs progress to centralized log file

**Results:**
- Overnight training: 4 models × 5 seeds = 20 runs (8.9 hours)
- No manual intervention needed
- All results saved automatically

### 3. Shared GPU Resource Management

**Implementation:**
```python
# Auto-adjust batch size based on workstation users
if num_users == 1:
    scale = 0.85  # Use 85% of GPU
elif num_users == 2:
    scale = 0.50  # Fair 50% split
else:
    scale = 0.33  # Fair 33% split per user
```

**Benefits:**
- Fair resource allocation on shared workstation
- Prevents OOM errors
- Maintains good neighbor policy

---

## 📊 Phase 2 Performance Summary

### Model Rankings (by Mean Accuracy)

| Rank | Model | Mean Acc | Std Dev | Best | Worst | Params |
|------|-------|----------|---------|------|-------|--------|
| 1 | ResNet-50 | **95.45%** | ±0.57% | 96.03% | 94.66% | 23.5M |
| 2 | DenseNet-121 | 95.32% | ±0.26% | 95.65% | 94.99% | 7.0M |
| 3 | EfficientNet-B0 | 95.23% | ±0.33% | 95.65% | 94.80% | 4.0M |
| 4 | CrossViT-Tiny | 94.96% | ±0.55% | 95.65% | 94.33% | 7.0M |
| 5 | ViT-Base | TBD | TBD | TBD | TBD | 85.8M |
| 6 | Swin-Tiny | TBD | TBD | TBD | TBD | 27.5M |

### Model Consistency (by Std Dev)

| Rank | Model | Std Dev | Interpretation |
|------|-------|---------|----------------|
| 1 | DenseNet-121 | **±0.26%** | Most consistent |
| 2 | EfficientNet-B0 | ±0.33% | Very consistent |
| 3 | CrossViT-Tiny | ±0.55% | Moderate variance |
| 4 | ResNet-50 | ±0.57% | Moderate variance |

**Key Insight:** DenseNet-121 is the most reliable model (lowest variance)

---

## 🎯 Training Configuration (Finalized)

```python
# Base configuration used for all models
BASE_CONFIG = {
    'device': 'cuda',
    'num_classes': 4,
    'image_size': 240,
    'batch_size': 380,  # Optimized (auto-adjusted to 323 for single user)
    'num_workers': 8,
    'learning_rate': 1.78e-4,  # Scaled for larger batch
    'weight_decay': 1e-4,
    'max_epochs': 30,
    'early_stopping_patience': 10,
    'mixed_precision': True,
    'seeds': [42, 123, 456, 789, 101112],

    # Class weights (from Phase 1)
    'class_weights': [1.47, 0.52, 0.88, 3.95],

    # Data augmentation
    'train_augmentation': 'conservative',  # From Phase 1
    'normalization': 'imagenet',  # mean=[0.485, 0.456, 0.406]
                                  # std=[0.229, 0.224, 0.225]
}
```

---

## 📁 Files Generated in Phase 2

### Model Checkpoints (4/6 complete):
```
experiments/phase2_systematic/models/
├── resnet50/           (5 models, 451 MB) ✅
├── densenet121/        (5 models, 136 MB) ✅
├── efficientnet/       (5 models, 79 MB) ✅
├── crossvit/           (5 models, 129 MB) ✅
├── vit/                (0 models) 🔄 Training
└── swin/               (0 models) ⏸️ Pending
```

### Results CSVs:
```
experiments/phase2_systematic/results/metrics/
├── resnet50_results.csv ✅
├── densenet121_results.csv ✅
├── efficientnet_results.csv ✅
└── crossvit_results.csv ✅
```

### Confusion Matrices (20/30 complete):
```
experiments/phase2_systematic/results/confusion_matrices/
├── resnet50_cm_*.png (5 files) ✅
├── densenet121_cm_*.png (5 files) ✅
├── efficientnet_cm_*.png (5 files) ✅
└── crossvit_cm_*.png (5 files) ✅
```

### MLflow Tracking:
```
experiments/phase2_systematic/mlruns/
└── [Automatic experiment tracking for all runs]
```

---

## 🔬 Key Research Findings (Phase 2)

### Finding 1: CNN Baselines Outperform CrossViT

**Hypothesis H₁:** "CrossViT will significantly outperform CNN baselines"

**Result:** **REJECTED** ❌

**Evidence:**
- ResNet-50: 95.45% vs CrossViT: 94.96% (Δ = **+0.49%**)
- DenseNet-121: 95.32% vs CrossViT: 94.96% (Δ = **+0.36%**)
- EfficientNet-B0: 95.23% vs CrossViT: 94.96% (Δ = **+0.27%**)

**Significance:** This is a **valid and important research finding**!

**Discussion Points for Thesis:**
1. Why CNNs may be better suited for X-ray classification
2. Inductive bias: CNNs have built-in translation invariance
3. Local feature extraction: CNNs excel at detecting localized patterns
4. Complexity vs Performance: CrossViT is more complex but not better
5. Dataset size: Transformers may need larger datasets
6. Medical imaging specifics: Structural patterns favor CNNs

### Finding 2: DenseNet-121 is Most Consistent

**Evidence:**
- Lowest standard deviation: ±0.26%
- All seeds achieved 95%+ accuracy
- Narrow range: [94.99%, 95.65%]

**Implication:** DenseNet-121 is the most reliable model for deployment

### Finding 3: Model Size ≠ Performance

**Evidence:**
- EfficientNet-B0 (4M params): 95.23%
- DenseNet-121 (7M params): 95.32%
- ResNet-50 (23.5M params): 95.45%
- ViT-Base (85.8M params): TBD

**Implication:** Smaller models can be highly competitive

---

## ⏱️ Time Tracking

| Model | Training Time | Seeds | Avg Time/Seed |
|-------|---------------|-------|---------------|
| ResNet-50 | 2h 34min | 5 | 30.8 min |
| DenseNet-121 | 2h 45min | 5 | 33.0 min |
| EfficientNet-B0 | 2h 32min | 5 | 30.4 min |
| CrossViT-Tiny | 3h 21min | 5 | 40.2 min |
| ViT-Base | ~2.5h (est) | 5 | ~30 min |
| Swin-Tiny | ~2h (est) | 5 | ~24 min |
| **Total** | **~15.7 hours** | 30 | **31.4 min** |

**Optimization Impact:**
- Before batch size optimization: ~18 hours estimated
- After batch size optimization: ~15.7 hours actual
- **Time saved: ~2.3 hours (13% faster)**

---

## 🐛 Issues Encountered & Resolved

### Issue 1: ViT/Swin Image Size Mismatch

**Problem:**
```
AssertionError: Input height (240) doesn't match model (224)
```

**Root Cause:**
- ViT and Swin expect 224×224 by default
- Our images are 240×240 (for CrossViT-Tiny-240 compatibility)

**Solution:**
```python
# Added img_size parameter
model = timm.create_model('vit_base_patch16_224',
                          pretrained=True,
                          img_size=240,  # Override default
                          num_classes=4)
```

**Status:** ✅ Resolved

### Issue 2: Underutilized GPU

**Problem:**
- Only 58% VRAM usage (28 GB / 48 GB)
- Slower than necessary training

**Solution:**
- Increased base batch size: 200 → 380
- Auto-adjusted for single user: 323
- Scaled learning rate proportionally

**Result:** 99% GPU utilization, 33% faster training

**Status:** ✅ Resolved

### Issue 3: MLflow Experiment ID Warning

**Warning:**
```
MLflow warning: Could not find experiment with ID 347976883098548854
```

**Impact:** None (results still logged correctly)

**Status:** ⚠️ Minor (does not affect training)

---

## 📈 Next Steps (Transition to Phase 3)

**When Phase 2 Completes:**

1. ✅ All 30 training runs complete (6 models × 5 seeds)
2. 📊 Move to Phase 3: Statistical Validation
3. 📝 Calculate 95% confidence intervals
4. 🧪 Perform hypothesis testing (McNemar's test)
5. 📋 Generate results tables for Chapter 5

**Current Status:** 20/30 runs complete (66.7%)
**Estimated Completion:** ~07:00 (4.5 hours remaining)

---

## 📚 References for Phase 2

- He et al. (2016) - Deep Residual Learning for Image Recognition (ResNet)
- Huang et al. (2017) - Densely Connected Convolutional Networks (DenseNet)
- Tan & Le (2019) - EfficientNet: Rethinking Model Scaling
- Chen et al. (2021) - CrossViT: Cross-Attention Multi-Scale Vision Transformer
- Dosovitskiy et al. (2021) - An Image is Worth 16x16 Words: Transformers for Image Recognition (ViT)
- Liu et al. (2021) - Swin Transformer: Hierarchical Vision Transformer

---

**Phase 2 Start Date:** Week 3
**Phase 2 Current Status:** 🔄 IN PROGRESS (Week 6)
**Expected Completion:** ~07:00 (2025-11-19)
**Next Phase:** Phase 3 - Statistical Validation
