# Phase 1: Exploration & Data Preparation

**Duration:** Week 1-2 (Completed)
**Goal:** Understand dataset + Get ONE baseline model working
**Status:** ✅ COMPLETED

---

## 📋 Phase Overview

Phase 1 focuses on setting up the environment, understanding the COVID-19 dataset, and validating that the training pipeline works end-to-end with at least one baseline model.

---

## ✅ Completed Tasks

### 1. Environment Setup (00_environment_setup.ipynb) ✅

**Completed:** Week 1, Day 1

**What Was Done:**
- Verified GPU: NVIDIA RTX 6000 Ada Generation (48 GB VRAM)
- Confirmed CUDA availability: PyTorch with CUDA 13.0
- Tested all dependencies (torch, torchvision, timm, opencv, pandas, etc.)
- Verified CrossViT model loading capability

**Key Outputs:**
```
GPU: NVIDIA RTX 6000 Ada Generation
VRAM: 48 GB
CUDA Version: 13.0
PyTorch Version: 2.x
All dependencies: ✅ Installed
```

**Files Created:**
- Environment verification logs
- Dependency list confirmation

---

### 2. Data Loading & Splitting (01_data_loading.ipynb) ✅

**Completed:** Week 1, Day 2

**What Was Done:**
- Loaded COVID-19 Radiography Dataset
- Total images: 21,165 chest X-rays (299×299 PNG, grayscale)
- Created stratified train/val/test split (80/10/10)
- Saved image paths to CSV files

**Dataset Statistics:**

| Split | Total Images | COVID | Normal | Lung Opacity | Viral Pneumonia |
|-------|-------------|-------|--------|--------------|-----------------|
| Train | 16,931 | 2,893 | 8,153 | 4,810 | 1,075 |
| Val | 2,117 | 362 | 1,020 | 601 | 134 |
| Test | 2,117 | 361 | 1,019 | 601 | 136 |
| **Total** | **21,165** | **3,616** | **10,192** | **6,012** | **1,345** |

**Class Imbalance Ratio:** 7.6:1 (Normal:Viral Pneumonia)

**Key Outputs:**
- `data/processed/train.csv` (16,931 rows)
- `data/processed/val.csv` (2,117 rows)
- `data/processed/test.csv` (2,117 rows)

**Reproducibility:**
- Random seed: 42
- Stratified split ensures class distribution maintained

---

### 3. Data Preprocessing & CLAHE (02_data_cleaning.ipynb) ✅

**Completed:** Week 1, Day 3

**What Was Done:**
- Applied CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Resized all images to 240×240 pixels
- Converted grayscale → RGB (for pretrained models)
- Implemented on-the-fly preprocessing (no saved preprocessed images)

**CLAHE Parameters:**
```python
clipLimit = 2.0
tileGridSize = (8, 8)
```

**Image Pipeline:**
```
Raw Image (299×299 grayscale)
    ↓
CLAHE Enhancement (clip=2.0, tile=8×8)
    ↓
Resize to 240×240
    ↓
Convert to RGB (3 channels)
    ↓
Normalize (ImageNet statistics)
    ↓
Ready for Training
```

**Why 240×240?**
- CrossViT-Tiny-240 (main model) requires 240×240 input
- Ensures fair comparison across all models
- Sufficient resolution for medical image analysis

**Key Design Decision:**
- **On-the-fly preprocessing** instead of saving preprocessed images
- Saves disk space (~21K images × 3 = ~60K files avoided)
- Allows flexible augmentation during training

---

### 4. Exploratory Data Analysis (03_eda.ipynb) ✅

**Completed:** Week 1, Day 4

**What Was Done:**
- Class distribution analysis with visualizations
- Pixel intensity statistics (before/after CLAHE)
- Statistical tests (ANOVA) to verify class separability
- Generated publication-ready figures for thesis

**Key Findings:**

**Class Distribution:**
- Severe imbalance: Normal (48.2%) vs Viral Pneumonia (6.4%)
- Calculated class weights for training: [1.47, 0.52, 0.88, 3.95]

**Pixel Intensity Analysis:**
- Raw images: Mean intensity varies significantly by class
- After CLAHE: Better contrast, more uniform histograms
- ANOVA F-statistic: Significant (p < 0.001) - classes are separable

**Figures Generated:**
1. Class distribution bar chart
2. Sample images from each class
3. Pixel intensity histograms (before/after CLAHE)
4. Box plots of intensity by class

**Statistical Validation:**
- One-way ANOVA confirms classes have distinct pixel distributions
- Supports feasibility of classification task

---

### 5. Baseline Model Test (04_baseline_test.ipynb) ✅

**Completed:** Week 2, Day 1

**What Was Done:**
- Created PyTorch Dataset and DataLoader
- Trained ResNet-50 as first baseline model
- Verified end-to-end training pipeline
- Tested on small subset first (1,000 images)
- Achieved >70% accuracy on validation set

**ResNet-50 Quick Test Results:**
```
Subset: 1,000 images
Epochs: 5
Batch Size: 32
Validation Accuracy: 78.5%
GPU Memory: 6.2 GB / 48 GB
Status: ✅ Pipeline working correctly
```

**Key Validations:**
- ✅ GPU memory within limits (<8 GB for safety)
- ✅ Data loading working correctly
- ✅ Training loop runs without errors
- ✅ Model checkpointing functional
- ✅ Mixed precision training enabled

**Training Configuration Finalized:**
```python
batch_size = 200  # Base (will be optimized in Phase 2)
learning_rate = 1e-4
weight_decay = 1e-4
max_epochs = 30
early_stopping_patience = 10
optimizer = Adam
scheduler = ReduceLROnPlateau
mixed_precision = True
```

---

### 6. Data Augmentation Strategy (05_augmentation_test.ipynb) ✅

**Completed:** Week 2, Day 2

**What Was Done:**
- Tested 3 augmentation strategies: None, Conservative, Aggressive
- Compared performance on validation set
- Selected conservative augmentation for Phase 2

**Augmentation Strategies Tested:**

**1. No Augmentation:**
- Val Acc: 76.2%
- Pros: Fast training
- Cons: Potential overfitting

**2. Conservative Augmentation (SELECTED):**
```python
transforms.RandomRotation(10)           # ±10° only
transforms.RandomHorizontalFlip(0.5)    # 50% horizontal flip
transforms.ColorJitter(brightness=0.1, contrast=0.1)  # Subtle
```
- Val Acc: 79.3%
- Pros: Improves generalization, realistic
- Cons: None significant

**3. Aggressive Augmentation:**
```python
transforms.RandomRotation(30)
transforms.RandomHorizontalFlip(0.5)
transforms.RandomVerticalFlip(0.3)      # ❌ Not anatomically valid
transforms.ColorJitter(brightness=0.3, contrast=0.3)
```
- Val Acc: 72.8%
- Cons: Degrades performance, unrealistic transformations

**Final Augmentation Pipeline (for Phase 2):**
```python
# Training augmentation
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Validation/Test (no augmentation)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

**Key Decision:**
- **Conservative augmentation** balances generalization and realism
- No vertical flips (anatomically incorrect for chest X-rays)
- Subtle transformations preserve medical image integrity

---

## 📊 Phase 1 Success Criteria (All Met ✅)

- [x] GPU and dependencies verified
- [x] Dataset loaded and split (21,165 images)
- [x] CLAHE preprocessing implemented
- [x] EDA completed with statistical analysis
- [x] ONE baseline model (ResNet-50) trains successfully
- [x] Training pipeline reproducible (seed=42)
- [x] GPU memory usage within limits (<8 GB tested)
- [x] Data augmentation strategy finalized

---

## 🔑 Key Decisions Made

### 1. Image Size: 240×240
**Rationale:**
- CrossViT-Tiny-240 requires this size
- Fair comparison across all models
- Sufficient resolution for medical imaging

### 2. On-the-fly CLAHE
**Rationale:**
- Saves disk space (no preprocessed images stored)
- Flexible for future experiments
- CPU preprocessing during data loading (GPU focused on training)

### 3. Conservative Augmentation
**Rationale:**
- Improves generalization without degrading accuracy
- Medically realistic transformations
- No vertical flips (anatomically incorrect)

### 4. Class Weights: [1.47, 0.52, 0.88, 3.95]
**Rationale:**
- Addresses severe class imbalance (7.6:1 ratio)
- Computed from class frequencies
- Used in CrossEntropyLoss

### 5. Batch Size Strategy
**Rationale:**
- Start with 200 (safe for 48 GB VRAM)
- Will optimize in Phase 2 based on GPU utilization
- Auto-adjust for shared workstation usage

---

## 📁 Files Generated in Phase 1

### Data Files:
```
data/processed/
├── train.csv          (16,931 rows)
├── val.csv            (2,117 rows)
└── test.csv           (2,117 rows)
```

### Notebooks:
```
notebooks/
├── 00_environment_setup.ipynb         ✅
├── 01_data_loading.ipynb              ✅
├── 02_data_cleaning.ipynb             ✅
├── 03_eda.ipynb                       ✅
├── 04_baseline_test.ipynb             ✅
└── 05_augmentation_test.ipynb         ✅
```

### Figures (for thesis):
```
results/figures/phase1/
├── class_distribution.png
├── sample_images_per_class.png
├── pixel_intensity_histograms.png
├── clahe_comparison.png
└── augmentation_comparison.png
```

---

## 🚀 Transition to Phase 2

**Ready to proceed with:**
- ✅ Validated training pipeline
- ✅ Optimized preprocessing
- ✅ Finalized augmentation strategy
- ✅ Class weights calculated
- ✅ Reproducibility ensured (seed=42)

**Next Phase:** Systematic training of all 6 models × 5 seeds (30 runs)

---

## 📝 Lessons Learned

1. **CLAHE significantly improves contrast** - visible difference in image quality
2. **Conservative augmentation works best** - aggressive augmentation hurts medical images
3. **240×240 resolution is sufficient** - no need for higher resolution
4. **On-the-fly preprocessing is efficient** - no disk space wasted
5. **Class imbalance is severe** - weighted loss is essential

---

## 📚 References for Phase 1

- Rahman et al. (2021) - COVID-19 Radiography Database
- Pizer et al. (1987) - CLAHE algorithm
- Perez & Wang (2017) - Data augmentation in medical imaging
- He et al. (2016) - ResNet architecture

---

**Phase 1 Completion Date:** Week 2
**Total Time:** ~2 weeks
**Next Phase:** Phase 2 - Systematic Experimentation
**Status:** ✅ COMPLETED - Ready for Phase 2
