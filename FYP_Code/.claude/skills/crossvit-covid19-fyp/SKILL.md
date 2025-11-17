---
name: crossvit-covid19-fyp
description: Complete context for TAR UMT Data Science FYP implementing CrossViT for COVID-19 chest X-ray classification. Use when working on Jupyter notebooks, code implementation, data analysis, model training, or any task related to Tan Ming Kai's final year project. This skill provides dataset specs, model architecture details, hardware constraints (RTX 4060 8GB VRAM), preprocessing parameters, baseline models, evaluation metrics, hypotheses, and coding guidelines for reproducible research following TAR UMT academic requirements.
---

# CrossViT COVID-19 FYP - Implementation Guide

## Project Overview

**Student:** Tan Ming Kai (24PMR12003)  
**Program:** Bachelor of Computer Science (Honours) in Data Science  
**University:** TAR UMT, Penang Branch, Malaysia  
**Academic Year:** 2025/26  
**Supervisor:** Angkay A/P Subramaniam

**Project Title:** Multi-Scale Vision Transformer (CrossViT) for COVID-19 Chest X-ray Classification Using Dual-Branch Architecture

**Core Philosophy:** COMPLETION OVER PERFECTION  
- Target: Pass with 50%+ (not publication-quality)
- Approach: Working code > Optimal code
- Timeline: Must complete within semester
- Constraint: RTX 4060 8GB VRAM (consumer hardware)

## Quick Reference

### Dataset Specifications
- **Name:** COVID-19 Radiography Database (Rahman et al., 2021)
- **Source:** Kaggle
- **Total Images:** 21,165 chest X-rays
- **Classes:** 4 (COVID-19, Normal, Lung Opacity, Viral Pneumonia)
- **Distribution:** 
  - COVID-19: 3,616 (17.1%)
  - Normal: 10,192 (48.2%)
  - Lung Opacity: 6,012 (28.4%)
  - Viral Pneumonia: 1,345 (6.3%)
- **Split:** 80% train (16,932) / 10% val (2,116) / 10% test (2,117)
- **Format:** PNG, 299×299 pixels, grayscale 8-bit
- **Imbalance Ratio:** 7.6:1 (Normal to Viral Pneumonia)

### Model Architecture
- **Primary Model:** CrossViT-Tiny from timm library
- **Input Size:** 240×240×3 (RGB)
- **Parameters:** ~7 million (fits in 8GB VRAM)
- **Patch Sizes:** 
  - Large branch: 16×16 patches, 384-768 dims
  - Small branch: 12×12 patches, 192-384 dims
- **Cross-Attention Layers:** K=3 multi-scale encoders
- **Complexity:** O(N) linear vs O(N²) quadratic

### Baseline Models (EXACTLY 5 Required)
1. **ResNet-50:** 25.6M params, CNN baseline
2. **DenseNet-121:** 8M params, dense connections
3. **EfficientNet-B0:** 5.3M params, compound scaling
4. **ViT-B/16:** 86M params, pure transformer baseline
5. **Swin-Tiny:** 28M params, hierarchical transformer

### Preprocessing Pipeline
```python
# Exact specifications from Chapter 4
1. Load image (grayscale/RGB handling)
2. CLAHE enhancement:
   - clip_limit = 2.0
   - tile_grid_size = (8, 8)  # Creates 64 contextual regions
3. Resize to 240×240 (CrossViT requirement)
4. Normalize to ImageNet stats:
   - mean = [0.485, 0.456, 0.406]
   - std = [0.229, 0.224, 0.225]
5. Data Augmentation (training only):
   - Random rotation: ±10°
   - Horizontal flip: 50% probability
   - Translation: ±5% in both axes
   - Brightness/contrast: ±10%
   - NO vertical flipping (anatomically incorrect)
   - NO aggressive elastic deformation
```

### Training Configuration
```python
# Hardware-optimized for RTX 4060 8GB
batch_size = 8  # Maximum 16 with gradient accumulation
gradient_accumulation_steps = 4  # Effective batch = 32
mixed_precision = True  # Use FP16 (automatic mixed precision)
optimizer = AdamW(lr=5e-5, weight_decay=0.05)
scheduler = CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
criterion = nn.CrossEntropyLoss(weight=[1.47, 0.52, 0.88, 3.95])  # Class weights
max_epochs = 50
early_stopping_patience = 15  # Monitor validation loss
seed = 42  # For reproducibility
```

### Evaluation Metrics (ALL with 95% CI)
**Primary Metrics:**
- Accuracy (overall and per-class)
- Precision, Recall, F1-Score (macro and weighted)
- AUC-ROC (one-vs-rest approach)
- Cohen's Kappa coefficient

**Medical Metrics:**
- Sensitivity/Specificity
- Positive/Negative Predictive Value
- Diagnostic Odds Ratio
- Youden's J statistic

**Statistical Validation:**
- 95% Confidence Intervals (bootstrap with 1000 iterations)
- Paired t-test (30 runs, α=0.05)
- McNemar's test (classification agreement)
- DeLong test (AUC comparison)
- Bonferroni correction (5 comparisons: α'=0.01)

### Research Hypotheses

**H₀ (Null):** No significant difference between CrossViT and CNN baselines (p≥0.05)

**H₁ (Primary):** CrossViT achieves significantly higher accuracy than CNN baselines (p<0.05)

**H₂ (Multi-scale):** Dual-branch processing improves accuracy by ≥5% vs single-scale

**H₃ (CLAHE):** Contrast enhancement improves performance by ≥2% vs no CLAHE

**H₄ (Augmentation):** Conservative augmentation improves generalization without degrading accuracy

### Hardware Constraints & Memory Management

**Available Resources:**
- GPU: NVIDIA RTX 4060 8GB VRAM
- CPU: AMD Ryzen 7, 32GB RAM
- Storage: NVMe SSD (fast data loading)
- OS: Ubuntu 24 (Linux environment)

**Critical Memory Tactics:**
```python
import torch

# 1. Clear cache frequently
torch.cuda.empty_cache()

# 2. Monitor memory usage
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

# 3. Use gradient checkpointing for large models
model.gradient_checkpointing_enable()

# 4. Delete unnecessary tensors
del outputs, loss
torch.cuda.empty_cache()

# 5. Use DataLoader with num_workers=4, pin_memory=True
train_loader = DataLoader(
    dataset, 
    batch_size=8, 
    num_workers=4, 
    pin_memory=True,
    persistent_workers=True  # Faster epoch transitions
)
```

## Notebook Development Guidelines

### Recommended Notebook Sequence
```
00_Environment_Setup.ipynb          # Verify GPU, packages, dependencies
01_Data_Exploration.ipynb           # EDA with visualizations (Chapter 4 evidence)
02_Data_Preprocessing.ipynb         # CLAHE optimization, class balance
03_Data_Augmentation.ipynb          # Test augmentation strategies
04_Baseline_Models.ipynb            # Train all 5 baselines
05_CrossViT_Training.ipynb          # Main model training
06_Results_Analysis.ipynb           # Statistical tests, 95% CI, hypothesis testing
07_Ablation_Studies.ipynb           # Test H2, H3, H4 hypotheses
08_Flask_Demo_Prep.ipynb            # Export model for web interface
```

### Mandatory Notebook Structure
Every notebook MUST include:

```python
"""
Notebook: XX_Name.ipynb
Purpose: [Clear single-sentence description]
Author: Tan Ming Kai (24PMR12003)
Date: 2025-XX-XX
FYP: CrossViT for COVID-19 Classification
Hardware: RTX 4060 8GB VRAM

Relates to: Chapter 4, Section X.X
Key Outputs: [List deliverables]
"""

# 1. REPRODUCIBILITY (ALWAYS FIRST)
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 2. IMPORTS (organized by category)
# Standard library
import os
from pathlib import Path

# Data science
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Computer Vision
import cv2
from PIL import Image
import timm

# 3. CONFIGURATION (single source of truth)
CONFIG = {
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir': '/path/to/covid19_radiography_database',
    'output_dir': './outputs',
    'batch_size': 8,
    'num_workers': 4,
    # ... all parameters here
}

# 4. HARDWARE VERIFICATION
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
```

### Publication-Quality Visualizations
```python
# Set publication style
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'

# Always label axes, add titles, legends
# Always save high-res versions for report
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
```

### Memory-Safe Training Loop Template
```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(loader):
        # Move to device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)  # More efficient
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Clear cache every N batches
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        # Memory monitoring (can be commented out in production)
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}: VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    return total_loss / len(loader)
```

## Critical Reminders

### What You MUST Do ✅
- Set seed=42 for ALL random operations (Python, NumPy, PyTorch, CUDA)
- Monitor GPU memory in every notebook
- Save all intermediate outputs (preprocessed data, trained models, metrics)
- Include proper error handling (file not found, CUDA OOM, etc.)
- Document every decision (why this parameter? why this approach?)
- Create visualizations for Chapter 4 (EDA plots, confusion matrices, ROC curves)
- Test code on SMALL subset first before full training
- Use tqdm progress bars for long operations
- Log all experiments (learning curves, metrics, timestamps)

### What You MUST NOT Do ❌
- Train without early stopping (waste of time/GPU)
- Use batch_size > 16 (will OOM on 8GB VRAM)
- Skip data validation (corrupted images cause crashes)
- Hardcode paths (use pathlib.Path for portability)
- Use too many workers (num_workers > 4 causes CPU overhead)
- Load entire dataset into RAM (use DataLoader)
- Save models without validation (check metrics first)
- Ignore warnings (fix them, they indicate issues)
- Submit code that doesn't run (test everything!)

### TAR UMT Academic Requirements
- Turnitin similarity must be <20%
- All code must be original or properly attributed
- Use APA 7th Edition citations in comments when using algorithms/methods from papers
- Keep Jupyter notebooks clean and well-documented (examiners will review)
- Include timing measurements (for performance claims in Chapter 5)
- Document all hyperparameter choices (needed for Chapter 4 justification)

## SDG Contributions

This project supports multiple UN Sustainable Development Goals:

**Primary: SDG 3 (Good Health and Well-being)**
- Target 3.3: Combat communicable diseases
- Impact: Rapid COVID-19 screening, 500+ patients/day processing

**Secondary: SDG 9 (Industry, Innovation, Infrastructure)**
- Target 9.5: Enhance scientific research
- Impact: Advancing AI in medical imaging for developing nations

**Tertiary: SDG 10 (Reduced Inequalities)**
- Target 10.2: Promote universal social inclusion
- Impact: Accessible diagnostics for rural/underserved areas in Malaysia

## Additional Resources

For detailed technical specifications, implementation details, and academic context:
- `references/technical_specs.md` - Complete technical specifications
- `references/academic_context.md` - Full academic background from Chapters 1-4

For utility scripts:
- `scripts/memory_monitor.py` - GPU memory tracking utility
- `scripts/check_dataset.py` - Validate dataset integrity
- `scripts/setup_env.py` - Environment verification script

## Quick Start Example

```python
# Minimal working example to verify setup
import torch
import timm

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load CrossViT-Tiny
model = timm.create_model('crossvit_tiny_240', pretrained=True, num_classes=4)
model = model.to(device)

# Test forward pass
dummy_input = torch.randn(1, 3, 240, 240).to(device)
with torch.no_grad():
    output = model(dummy_input)
print(f"Output shape: {output.shape}")  # Should be [1, 4]

print("✅ Setup verified! Ready to start FYP implementation.")
```

## Success Criteria

**You PASS when:**
- CrossViT model trains successfully and achieves >85% accuracy
- All 5 baselines tested (even if accuracy is suboptimal)
- Statistical tests completed (paired t-test, McNemar, DeLong)
- 95% CI reported for all metrics
- Hypothesis H₁ validated (p<0.05)
- Basic Flask interface works
- All notebooks run without errors
- Report submitted on time

**You DON'T need:**
- 95%+ accuracy (85-90% is sufficient for pass)
- Beautiful web interface (basic functional demo is enough)
- Publication-quality code (working code is sufficient)
- Perfect hyperparameters (reasonable defaults are fine)

Remember: **DONE > PERFECT**. This is about graduation, not publication.
