# Technical Specifications Reference

## Complete Dataset Information

### COVID-19 Radiography Database Details

**Source:** Rahman et al. (2021), Qatar University  
**Kaggle Link:** [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)  
**Version:** Continuously updated since 2020  
**Total Images:** 21,165 chest X-rays  
**Image Format:** PNG files, converted from DICOM  
**Original Resolution:** 299×299 pixels  
**Color Space:** Grayscale 8-bit (256 intensity levels)  
**File Size:** ~15GB total dataset

### Class Distribution Analysis

| Class | Count | Percentage | Train | Val | Test |
|-------|-------|------------|-------|-----|------|
| COVID-19 | 3,616 | 17.1% | 2,893 | 362 | 361 |
| Normal | 10,192 | 48.2% | 8,154 | 1,019 | 1,019 |
| Lung Opacity | 6,012 | 28.4% | 4,810 | 601 | 601 |
| Viral Pneumonia | 1,345 | 6.3% | 1,075 | 134 | 136 |
| **Total** | **21,165** | **100%** | **16,932** | **2,116** | **2,117** |

**Imbalance Ratios:**
- Normal:Viral Pneumonia = 7.58:1
- Normal:COVID-19 = 2.82:1
- Lung Opacity:Viral Pneumonia = 4.47:1

### Class Weight Calculation
```python
from sklearn.utils.class_weight import compute_class_weight

# Class weights for nn.CrossEntropyLoss
class_weights = [1.47, 0.52, 0.88, 3.95]  # [COVID, Normal, Opacity, Viral]

# Formula: weight = n_samples / (n_classes * n_samples_per_class)
# COVID-19: 21165 / (4 * 3616) = 1.47
# Normal: 21165 / (4 * 10192) = 0.52
# Lung Opacity: 21165 / (4 * 6012) = 0.88
# Viral Pneumonia: 21165 / (4 * 1345) = 3.95
```

## Complete Model Specifications

### CrossViT-Tiny Architecture

**Model Name:** `crossvit_tiny_240` (from timm library)  
**Paper:** Chen et al. (2021), "CrossViT: Cross-Attention Multi-Scale Vision Transformer"  
**Input Resolution:** 240×240×3 (RGB, even for grayscale input)  
**Total Parameters:** 7,011,440 (~7M)  
**FLOPs:** ~1.7 GFLOPs (efficient for real-time inference)  
**Memory Footprint:** ~2.5GB VRAM during inference, ~6GB during training

**Architecture Details:**
```python
# Dual-branch structure
Small Branch:
  - Patch size: 12×12 pixels
  - Embedding dim: 96
  - Number of patches: (240/12)² = 400 patches
  - Transformer depth: 4 blocks
  - MLP ratio: 3
  - Attention heads: 3

Large Branch:
  - Patch size: 16×16 pixels
  - Embedding dim: 128
  - Number of patches: (240/16)² = 225 patches
  - Transformer depth: 4 blocks
  - MLP ratio: 3
  - Attention heads: 4

Cross-Attention Fusion:
  - Fusion layers: 3 (after blocks 1, 2, 3)
  - Cross-attention heads: 3
  - Fusion method: Bi-directional cross-attention

Classification Head:
  - Method: Concatenate [CLS] tokens from both branches
  - Linear layer: (96+128) → 4 classes
  - Activation: Softmax
```

### Baseline Model Specifications

**1. ResNet-50**
```python
Model: torchvision.models.resnet50(pretrained=True)
Input: 224×224×3
Parameters: 25,557,032 (~25.6M)
Depth: 50 layers
Architecture: Conv→BN→ReLU→MaxPool→4 Residual Stages→AvgPool→FC
Pre-training: ImageNet-1K
Modification: Replace final FC layer for 4 classes
VRAM: ~4GB during training with batch_size=8
```

**2. DenseNet-121**
```python
Model: torchvision.models.densenet121(pretrained=True)
Input: 224×224×3
Parameters: 7,978,856 (~8M)
Depth: 121 layers
Architecture: Conv→BN→ReLU→MaxPool→4 Dense Blocks→FC
Dense connectivity: Each layer receives feature maps from all previous layers
Pre-training: ImageNet-1K
Modification: Replace classifier for 4 classes
VRAM: ~3.5GB during training with batch_size=8
```

**3. EfficientNet-B0**
```python
Model: timm.create_model('efficientnet_b0', pretrained=True)
Input: 224×224×3
Parameters: 5,288,548 (~5.3M)
Compound scaling: Balanced depth, width, resolution
Architecture: MBConv blocks with squeeze-and-excitation
Pre-training: ImageNet-1K
Modification: Replace classifier for 4 classes
VRAM: ~3GB during training with batch_size=8
```

**4. Vision Transformer Base (ViT-B/16)**
```python
Model: timm.create_model('vit_base_patch16_224', pretrained=True)
Input: 224×224×3
Parameters: 86,567,656 (~86M)
Patch size: 16×16 pixels
Architecture: 12 Transformer encoder layers
Embedding dim: 768
Attention heads: 12
Pre-training: ImageNet-21K
Modification: Replace head for 4 classes
VRAM: ~7GB during training with batch_size=8 (TIGHT!)
NOTE: May need batch_size=4 with gradient accumulation
```

**5. Swin Transformer Tiny**
```python
Model: timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
Input: 224×224×3
Parameters: 28,288,354 (~28M)
Patch size: 4×4 pixels
Window size: 7×7 for local attention
Architecture: 4 stages with shifted window mechanism
Pre-training: ImageNet-1K
Modification: Replace head for 4 classes
VRAM: ~5GB during training with batch_size=8
```

## CLAHE Preprocessing Details

### Contrast Limited Adaptive Histogram Equalization

**Library:** OpenCV (cv2.createCLAHE)  
**Purpose:** Enhance local contrast without over-amplifying noise

**Parameters:**
```python
clahe = cv2.createCLAHE(
    clipLimit=2.0,      # Contrast limiting threshold
    tileGridSize=(8,8)  # Size of grid for histogram equalization
)

# Creates 64 contextual regions (8×8 grid)
# Each region: 240/8 = 30 pixels per tile
```

**Parameter Justification:**
- `clipLimit=2.0`: Conservative value prevents over-enhancement of noise
  - Range typically 1.0-4.0
  - Lower = less contrast, higher = more contrast
  - 2.0 optimal for chest X-rays (validated in literature)
  
- `tileGridSize=(8,8)`: Balance between local and global adaptation
  - Smaller tiles = more local adaptation (more aggressive)
  - Larger tiles = more global adaptation (less aggressive)
  - 8×8 standard for medical images (30×30 pixel tiles)

**Application Process:**
```python
def apply_clahe(image):
    """Apply CLAHE to grayscale image"""
    # Ensure 8-bit grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # Apply CLAHE
    enhanced = clahe.apply(image)
    
    return enhanced
```

**Expected Improvements:**
- Contrast Index: +15.3% improvement
- Structural Similarity (SSIM): 0.89 (preserves structure)
- Ground-glass opacity visibility: Enhanced
- Consolidation edges: Sharper

## Data Augmentation Specifications

### Training Augmentation Pipeline

**Library:** Albumentations 1.3.1  
**Apply Probability:** 0.5 per augmentation (except compose)

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    # Geometric transformations
    A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05,      # ±5% translation
        scale_limit=0.0,       # No scaling
        rotate_limit=0,        # Rotation handled above
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=0.5
    ),
    
    # Photometric transformations
    A.RandomBrightnessContrast(
        brightness_limit=0.1,   # ±10% brightness
        contrast_limit=0.1,     # ±10% contrast
        p=0.5
    ),
    
    # Final processing
    A.Resize(240, 240),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

# Validation/Test transform (no augmentation)
val_transform = A.Compose([
    A.Resize(240, 240),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])
```

**Explicitly EXCLUDED Augmentations:**
- ❌ Vertical flipping (anatomically incorrect)
- ❌ Elastic deformation (distorts pathology)
- ❌ Cutout/CoarseDropout (may remove diagnostic regions)
- ❌ Gaussian noise (amplifies artifacts)
- ❌ Motion blur (unrealistic for X-rays)
- ❌ JPEG compression (dataset already PNG)

## Training Hyperparameters

### Optimizer Configuration
```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

optimizer = AdamW(
    model.parameters(),
    lr=5e-5,                    # Initial learning rate
    betas=(0.9, 0.999),        # Momentum parameters
    eps=1e-8,                   # Numerical stability
    weight_decay=0.05,          # L2 regularization
    amsgrad=False
)

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,                     # Initial restart period (epochs)
    T_mult=2,                   # Period multiplication after restart
    eta_min=1e-7                # Minimum learning rate
)

# Learning rate schedule:
# Epochs 1-10: Cosine decay from 5e-5 to 1e-7
# Epochs 11-30: Cosine decay from 5e-5 to 1e-7 (20 epochs)
# Epochs 31-70: Cosine decay from 5e-5 to 1e-7 (40 epochs)
```

### Loss Function
```python
import torch.nn as nn

# Weighted Cross-Entropy for class imbalance
class_weights = torch.tensor([1.47, 0.52, 0.88, 3.95]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Alternative: Focal Loss (optional, for experimentation)
# from focal_loss import FocalLoss
# criterion = FocalLoss(alpha=class_weights, gamma=2.0)
```

### Training Configuration
```python
CONFIG = {
    # Training
    'max_epochs': 50,
    'early_stopping_patience': 15,
    'early_stopping_delta': 0.001,  # Minimum improvement
    
    # Batch processing
    'batch_size': 8,
    'gradient_accumulation_steps': 4,
    'effective_batch_size': 32,  # 8 * 4
    
    # Mixed precision
    'use_amp': True,
    'amp_dtype': torch.float16,
    
    # Data loading
    'num_workers': 4,
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 2,
    
    # Gradient clipping
    'max_grad_norm': 1.0,
    
    # Checkpoint saving
    'save_best_only': True,
    'save_frequency': 5,  # Save every 5 epochs
    
    # Logging
    'log_interval': 50,  # Log every 50 batches
    'tensorboard': True,
    
    # Reproducibility
    'seed': 42,
    'cudnn_deterministic': True,
    'cudnn_benchmark': False
}
```

## Memory Optimization Strategies

### VRAM Management for 8GB GPU

**1. Batch Size Selection**
```python
# Safe batch sizes for RTX 4060 8GB:
batch_sizes = {
    'crossvit_tiny_240': 8,      # Safe
    'resnet50': 16,              # Safe
    'densenet121': 16,           # Safe
    'efficientnet_b0': 16,       # Safe
    'vit_base_patch16_224': 4,   # Tight! Use gradient accumulation
    'swin_tiny': 8               # Safe
}

# If OOM occurs, reduce by half and double gradient_accumulation_steps
```

**2. Gradient Accumulation**
```python
# Simulate larger batch size without memory overhead
optimizer.zero_grad()
for i, (inputs, labels) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss = loss / gradient_accumulation_steps  # Scale loss
    loss.backward()
    
    if (i + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**3. Mixed Precision Training**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, labels in train_loader:
    optimizer.zero_grad()
    
    # Mixed precision forward pass
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
    # Scaled backward pass
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

**4. Gradient Checkpointing** (for very large models)
```python
# Enable for ViT-Base if still OOM
model.set_grad_checkpointing(enable=True)

# Or manually in forward pass:
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(x):
    return checkpoint(model.blocks, x)
```

## Evaluation Metrics Formulas

### Primary Classification Metrics

**Accuracy:**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision (per class):**
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall / Sensitivity:**
$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1-Score:**
$$F1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Macro-Average (equal weight per class):**
$$\text{Macro-F1} = \frac{1}{C} \sum_{i=1}^{C} F1_i$$

**Weighted-Average (weight by support):**
$$\text{Weighted-F1} = \frac{1}{N} \sum_{i=1}^{C} n_i \cdot F1_i$$

### Medical Diagnostic Metrics

**Specificity:**
$$\text{Specificity} = \frac{TN}{TN + FP}$$

**Positive Predictive Value (PPV):**
$$\text{PPV} = \frac{TP}{TP + FP}$$

**Negative Predictive Value (NPV):**
$$\text{NPV} = \frac{TN}{TN + FN}$$

**Cohen's Kappa:**
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
where $p_o$ = observed agreement, $p_e$ = expected agreement by chance

### Statistical Tests

**Paired t-test statistic:**
$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$
where $\bar{d}$ = mean difference, $s_d$ = standard deviation of differences, $n$ = sample size

**McNemar's test statistic:**
$$\chi^2 = \frac{(b - c)^2}{b + c}$$
where $b$ = cases where model A correct but B wrong, $c$ = vice versa

**95% Confidence Interval (bootstrap):**
$$CI_{95\%} = [P_{2.5}, P_{97.5}]$$
where $P_{2.5}$ and $P_{97.5}$ are 2.5th and 97.5th percentiles of bootstrap distribution

## Python Environment Setup

### Required Packages
```bash
# Core deep learning
torch==2.0.1
torchvision==0.15.2
timm==0.9.2

# Computer vision
opencv-python==4.8.0.74
albumentations==1.3.1
scikit-image==0.21.0

# Data science
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2

# Scientific computing
scipy==1.11.0
scikit-learn==1.3.0

# Utilities
tqdm==4.65.0
pillow==10.0.0

# Logging
tensorboard==2.13.0
wandb==0.15.5  # Optional

# Statistical analysis
statsmodels==0.14.0
pingouin==0.5.3  # For advanced stats

# Flask (for demo)
flask==2.3.2
flask-cors==4.0.0
gunicorn==21.2.0
```

### Installation Commands
```bash
# Create virtual environment
python -m venv fyp_env
source fyp_env/bin/activate  # Linux/Mac
# fyp_env\Scripts\activate  # Windows

# Install PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining packages
pip install timm opencv-python albumentations scikit-image
pip install numpy pandas matplotlib seaborn scipy scikit-learn
pip install tqdm pillow tensorboard statsmodels pingouin
pip install flask flask-cors gunicorn

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Directory Structure

```
crossvit-covid19-fyp/
├── data/
│   ├── raw/                          # Original dataset from Kaggle
│   │   ├── COVID/
│   │   ├── Normal/
│   │   ├── Lung_Opacity/
│   │   └── Viral_Pneumonia/
│   ├── processed/                    # CLAHE-enhanced images
│   └── splits/                       # Train/val/test splits
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── notebooks/
│   ├── 00_Environment_Setup.ipynb
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Data_Augmentation.ipynb
│   ├── 04_Baseline_Models.ipynb
│   ├── 05_CrossViT_Training.ipynb
│   ├── 06_Results_Analysis.ipynb
│   ├── 07_Ablation_Studies.ipynb
│   └── 08_Flask_Demo_Prep.ipynb
│
├── src/
│   ├── __init__.py
│   ├── dataset.py                    # PyTorch Dataset class
│   ├── models.py                     # Model definitions
│   ├── preprocessing.py              # CLAHE, augmentation
│   ├── training.py                   # Training loops
│   ├── evaluation.py                 # Metrics, statistical tests
│   └── utils.py                      # Helper functions
│
├── outputs/
│   ├── figures/                      # Plots for report
│   ├── models/                       # Saved checkpoints
│   ├── logs/                         # TensorBoard logs
│   └── results/                      # CSV files with metrics
│
├── flask_app/
│   ├── app.py                        # Flask application
│   ├── templates/                    # HTML templates
│   ├── static/                       # CSS, JS, images
│   └── model/                        # Exported model
│
├── requirements.txt
├── README.md
└── .gitignore
```

This structure separates concerns and maintains organization throughout the FYP development.
