# FYP Project - Ready for Remote Desktop Transfer

**Date:** 2025-11-13
**Student:** Tan Ming Kai (24PMR12003)
**Project:** CrossViT COVID-19 Classification

---

## Cleanup Summary

### Before Cleanup: 16.0 GB
### After Cleanup: 2.4 GB
### Space Saved: 13.6 GB (85% reduction)

---

## What Was Deleted

| Item | Size | Reason |
|------|------|--------|
| USB transfer files | 6.4 GB | Not needed for remote desktop |
| Python virtual environment | 5.5 GB | Will recreate on workstation |
| Zip_file folder | 778 MB | Redundant dataset archive |
| Raw dataset | 877 MB | Have processed data |
| Cache files (__pycache__) | 193 MB | Auto-regenerated |
| Training log | 17 MB | Data in notebooks/MLflow |
| CrossViT checkpoint | 26 MB | Will retrain on workstation |
| Temp files | <1 KB | Windows temp files |

---

## What Was Kept (2.4 GB)

### Essential Data (2.0 GB)
- ✅ data/processed/clahe_enhanced/ - **21,165 images**
- ✅ data/processed/*.csv - **7 CSV files** (train/val/test splits)

### Trained Models (451 MB)
- ✅ models/resnet50_best_seed42.pth
- ✅ models/resnet50_best_seed123.pth
- ✅ models/resnet50_best_seed456.pth
- ✅ models/resnet50_best_seed789.pth
- ✅ models/resnet50_best_seed101112.pth

### All Notebooks (11 MB)
- ✅ 00_environment_setup.ipynb
- ✅ 01_data_loading.ipynb
- ✅ 02_data_cleaning.ipynb
- ✅ 03_eda.ipynb
- ✅ 04_baseline_test.ipynb
- ✅ 04_baseline_test_FULL.ipynb
- ✅ 06_crossvit_training.ipynb (PRIMARY MODEL)
- ✅ 07_resnet50_training.ipynb
- ✅ 08_densenet121_training.ipynb
- ✅ 09_efficientnet_training.ipynb
- ✅ 10_vit_training.ipynb
- ✅ 11_swin_training.ipynb
- ✅ 07_resnet50_training.py
- ✅ notebooks/mlruns/ (MLflow tracking)

### Results (4.6 MB)
- ✅ ResNet-50 confusion matrices (6 images)
- ✅ Training history plots
- ✅ resnet50_results.csv
- ✅ EDA figures (from Phase 1)

### Source Code & Documentation (121 KB)
- ✅ src/ folder (data_processing.py, features.py, models.py)
- ✅ CLAUDE.md
- ✅ requirements.txt
- ✅ WORKSTATION_TRANSFER_GUIDE.md
- ✅ NOTEBOOK_VALIDATION_REPORT.md
- ✅ All other .md documentation
- ✅ .claude/ skills directory

---

## Verification Results

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Processed Images | 21,165 | 21,165 | ✅ PASS |
| CSV Files | 7 | 7 | ✅ PASS |
| ResNet-50 Models | 5 | 5 | ✅ PASS |
| Notebooks | 13 | 13 | ✅ PASS |
| Documentation | 9+ | 9 | ✅ PASS |
| Source Code | 3+ | 4 | ✅ PASS |
| Results Files | 8+ | 8 | ✅ PASS |

**ALL CHECKS PASSED** ✅

---

## Remote Desktop Transfer Instructions

### Step 1: Connect to Lab Workstation

1. Connect to **AINexus24** Wi-Fi
   - Password: `12345678`

2. Open **Remote Desktop Connection**
   - Press `Win + R`, type `mstsc`, press Enter

3. Connect to workstation
   - Computer: `metacode1` or `metacode2`
   - Click "Connect"

4. Login credentials:
   - Username: `focs1` or `focs2` or `focs3`
   - Password: `focs123`

### Step 2: Copy Project Files

**On your laptop:**
1. Navigate to: `D:\Users\USER\Documents\GitHub\fyp-project\FYP_Code`

**In Remote Desktop:**
1. Open File Explorer
2. Navigate to `\\tsclient\D\Users\USER\Documents\GitHub\fyp-project\`
3. Copy entire `FYP_Code` folder to workstation drive (e.g., `C:\Users\focs1\`)
4. Wait for transfer to complete (~2.4 GB, estimated 5-15 minutes depending on network)

**Alternative - Direct Drive Mapping:**
1. Your local drives appear as `\\tsclient\C`, `\\tsclient\D`, etc. in Remote Desktop
2. Simply copy from `\\tsclient\D\...\FYP_Code` to local workstation folder

### Step 3: Setup Environment on Workstation

```bash
# Navigate to project folder
cd C:\Users\focs1\FYP_Code

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional packages
pip install timm mlflow

# Verify GPU available
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

# Verify data integrity
python -c "import os; print('Images:', len([f for f in os.listdir('data/processed/clahe_enhanced/train') if f.endswith('.png')]))"
```

**Expected Output:**
```
CUDA Available: True
GPU: [Workstation GPU name]
Images: [Should show ~16,000+ training images]
```

### Step 4: Start Training

**Option A: Run in Jupyter Notebook (Recommended for monitoring)**
```bash
cd notebooks
jupyter notebook
# Open 06_crossvit_training.ipynb and run all cells
```

**Option B: Run as Python script (Background training)**
```bash
cd notebooks
jupyter nbconvert --to script 06_crossvit_training.ipynb
python 06_crossvit_training.py > crossvit_training.log 2>&1
```

**Training Order:**
1. **06_crossvit_training.ipynb** - PRIMARY MODEL (5 seeds, ~6-8 hours)
2. **08_densenet121_training.ipynb** - Baseline 2 (5 seeds, ~3-4 hours)
3. **09_efficientnet_training.ipynb** - Baseline 3 (5 seeds, ~3-4 hours)
4. **10_vit_training.ipynb** - Baseline 4 (5 seeds, ~5-7 hours)
5. **11_swin_training.ipynb** - Baseline 5 (5 seeds, ~4-6 hours)

**Note:** ResNet-50 already trained (95.49% ± 0.33%)

---

## Lab Guidelines Reminders

1. **Training Time:** Can run overnight (as per lab guidelines)
2. **Lab Hours:** 1:30 PM - 3:00 PM during semester break
3. **Data Cleanup:** Remember to delete all your data after training completes
4. **Return Next Day:** If training runs overnight, return next day to retrieve results and clean up

---

## Expected Training Time on Workstation

**With Better GPU (estimated):**
- CrossViT: 6-8 hours (5 seeds)
- DenseNet-121: 3-4 hours (5 seeds)
- EfficientNet-B0: 3-4 hours (5 seeds)
- ViT-Base/16: 5-7 hours (5 seeds)
- Swin-Tiny: 4-6 hours (5 seeds)

**Total:** ~20-30 hours for all 5 remaining models

**If workstation has faster GPU than RTX 4060:**
- Can increase batch sizes for 2-3x speedup
- Update `batch_size` in notebook CONFIG sections

---

## Troubleshooting

### If GPU not detected:
```bash
# Check CUDA installation
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print(torch.version.cuda)"
```

### If out of memory:
- Reduce `batch_size` in notebook CONFIG
- Enable mixed precision (already enabled in notebooks)

### If MLflow not working:
```bash
# MLflow is optional - notebooks will run without it
# To install:
pip install mlflow
```

---

## What to Do After Training

1. **Collect Results:**
   - Model checkpoints: `models/`
   - Confusion matrices: `results/`
   - MLflow data: `notebooks/mlruns/`

2. **Copy Back to Your Laptop:**
   - Use Remote Desktop file sharing (reverse direction)
   - Copy `models/`, `results/`, `notebooks/mlruns/` back

3. **Clean Up Workstation:**
   - Delete entire `FYP_Code` folder
   - Clear any other personal data
   - As per lab rule: Students who don't clean up will be banned

---

## Project is Ready! ✅

- All unnecessary files removed
- All essential files verified
- Size optimized for fast remote desktop transfer
- GitHub backup available: `https://github.com/Ming-Kai-LC/fyp-project.git`

**Good luck with your training tomorrow!**
