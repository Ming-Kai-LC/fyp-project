# Workstation Transfer Guide

**Project:** TAR UMT FYP - CrossViT COVID-19 Classification
**From:** Laptop (RTX 4060 8GB)
**To:** Workstation (High-performance GPU)

---

## 📋 What You Need to Transfer

### ✅ Essential Files (MUST transfer)
```
FYP_Code/
├── notebooks/                    [~50 MB]
│   ├── 06_crossvit_training.ipynb
│   ├── 07_resnet50_training.ipynb
│   ├── 08_densenet121_training.ipynb
│   ├── 09_efficientnet_training.ipynb
│   ├── 10_vit_training.ipynb
│   └── 11_swin_training.ipynb
│
├── data/processed/               [~500 MB - 2 GB]
│   ├── train_processed.csv
│   ├── val_processed.csv
│   ├── test_processed.csv
│   └── clahe_enhanced/           [21,165 images]
│
├── models/                       [~500 MB if keeping ResNet-50]
│   └── resnet50_best_seed*.pth   [5 files - optional]
│
├── results/                      [~10 MB]
│   └── resnet50_*.*              [ResNet-50 results - optional]
│
├── requirements.txt              [Required]
├── CLAUDE.md                     [Documentation]
├── NOTEBOOK_VALIDATION_REPORT.md [Just created]
└── PHASE2_SETUP.md              [Documentation]
```

### ❌ Skip These (save time/space)
```
.git/                   [Can re-clone from GitHub]
venv/                   [Recreate on workstation]
__pycache__/           [Will regenerate]
.ipynb_checkpoints/    [Jupyter temp files]
notebooks/mlruns/      [MLflow tracking - will recreate]
```

**Total Transfer Size:** ~1-3 GB (depending on what you include)

---

## 🚀 Transfer Methods (Choose One)

### Method 1: Git + Data Sync (RECOMMENDED) ⭐

**Best for:** Clean transfer, version control

**Step 1: Commit and push current work**
```bash
# On laptop
cd D:\Users\USER\Documents\GitHub\fyp-project\FYP_Code

# Check status
git status

# Add new notebooks
git add notebooks/08_densenet121_training.ipynb
git add notebooks/09_efficientnet_training.ipynb
git add notebooks/10_vit_training.ipynb
git add notebooks/11_swin_training.ipynb
git add NOTEBOOK_VALIDATION_REPORT.md
git add WORKSTATION_TRANSFER_GUIDE.md

# Commit
git commit -m "Add Phase 2 baseline training notebooks (08-11) and validation report

- Created DenseNet-121, EfficientNet-B0, ViT-Base/16, Swin-Tiny notebooks
- All notebooks validated and ready for workstation execution
- Added comprehensive validation report
- ResNet-50 training complete (95.49% ± 0.33%)

Ready for Phase 2 systematic experimentation on workstation."

# Push to GitHub
git push origin main
```

**Step 2: On workstation, clone repository**
```bash
# On workstation
cd ~/projects  # or wherever you want
git clone https://github.com/YOUR_USERNAME/fyp-project.git
cd fyp-project/FYP_Code
```

**Step 3: Transfer data separately** (data too large for Git)

Choose one sub-method:

**Option A: USB Drive**
```bash
# Copy from laptop to USB
# Copy these folders:
- data/processed/
- models/ (if keeping ResNet-50 results)
- results/ (optional)

# Then copy from USB to workstation
```

**Option B: Network Share (if on same network)**
```bash
# On workstation (if laptop accessible via network)
scp -r user@laptop-ip:/path/to/FYP_Code/data/processed ~/projects/fyp-project/FYP_Code/data/
```

**Option C: Cloud Storage** (OneDrive, Google Drive)
```bash
# Zip data folder on laptop
# Upload to cloud
# Download on workstation
```

**Pros:**
- ✅ Clean, organized
- ✅ Version control maintained
- ✅ Easy to sync changes
- ✅ Best practice

**Cons:**
- ❌ Two-step process (code + data)

---

### Method 2: Complete USB/External Drive Transfer

**Best for:** Simple, all-in-one transfer

**On Laptop:**
```bash
# 1. Create clean copy (exclude unnecessary files)
# In File Explorer, copy these to USB:

FYP_Code/
├── notebooks/        [Copy all .ipynb files]
├── data/processed/   [Copy entire folder]
├── models/          [Optional - ResNet-50 results]
├── results/         [Optional]
├── requirements.txt
├── CLAUDE.md
├── *.md files
```

**On Workstation:**
```bash
# Copy from USB to workstation location
cp -r /media/usb/FYP_Code ~/projects/
```

**Pros:**
- ✅ Simple, straightforward
- ✅ Everything in one transfer
- ✅ No internet needed

**Cons:**
- ❌ No version control
- ❌ Slower for large data

---

### Method 3: Network Transfer (Same Network)

**Best for:** Fast transfer if laptop and workstation on same network

**Prerequisites:**
- Both machines on same network
- Know workstation IP address

**On Workstation (prepare to receive):**
```bash
# Install rsync if not present
sudo apt-get install rsync

# Note your IP address
ip addr show
```

**On Laptop (transfer files):**
```bash
# Transfer entire project (Windows using WSL or PowerShell with SSH)
rsync -avz --progress D:/Users/USER/Documents/GitHub/fyp-project/FYP_Code/ user@workstation-ip:~/projects/FYP_Code/

# Or use SCP
scp -r D:/Users/USER/Documents/GitHub/fyp-project/FYP_Code user@workstation-ip:~/projects/
```

**Pros:**
- ✅ Fast (network speed)
- ✅ Can resume interrupted transfers
- ✅ Preserves file permissions

**Cons:**
- ❌ Requires network setup
- ❌ Need SSH access

---

### Method 4: Cloud Storage (Google Drive, OneDrive, Dropbox)

**Best for:** No USB drive available, working remotely

**Steps:**
1. Compress data folder:
   ```bash
   # On laptop (in PowerShell or WSL)
   cd D:\Users\USER\Documents\GitHub\fyp-project
   tar -czf FYP_Code_data.tar.gz FYP_Code/data/processed/
   ```

2. Upload to cloud:
   - Upload `FYP_Code_data.tar.gz` (~1-2 GB compressed)
   - Upload notebook files separately (small)

3. Download on workstation:
   ```bash
   # Download and extract
   tar -xzf FYP_Code_data.tar.gz
   ```

**Pros:**
- ✅ Access from anywhere
- ✅ Backup copy in cloud

**Cons:**
- ❌ Slow upload/download
- ❌ Storage limits
- ❌ Internet required

---

## 🔧 Setup on Workstation (After Transfer)

### 1. Verify Files Transferred

```bash
cd ~/projects/FYP_Code  # or wherever you transferred

# Check structure
ls -la

# Verify data
ls data/processed/
ls data/processed/clahe_enhanced/ | wc -l  # Should show 21,165

# Verify notebooks
ls notebooks/*.ipynb
```

**Expected output:**
```
06_crossvit_training.ipynb
07_resnet50_training.ipynb
08_densenet121_training.ipynb
09_efficientnet_training.ipynb
10_vit_training.ipynb
11_swin_training.ipynb
```

### 2. Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows if workstation is Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install from requirements.txt
pip install -r requirements.txt

# Install additional packages for Phase 2
pip install timm mlflow

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 4. Test Model Loading

```bash
# Quick test
python << EOF
import torch
import timm
import mlflow

# Test model loading
model = timm.create_model('crossvit_tiny_240', pretrained=True, num_classes=4)
print(f"CrossViT loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# Check GPU
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
EOF
```

**Expected output:**
```
CrossViT loaded: 7,xxx,xxx parameters
GPU: [Your GPU name]
VRAM: [XX.XX] GB
```

### 5. Verify Data Paths

Open a notebook and run first few cells:
```bash
cd notebooks
jupyter notebook  # or jupyter lab
```

Test that data loads correctly:
```python
import pandas as pd
from pathlib import Path

CSV_DIR = Path("../data/processed")
train_df = pd.read_csv(CSV_DIR / "train_processed.csv")
print(f"Train: {len(train_df):,} images")
# Should show: Train: 16,931 images
```

---

## 🎯 Ready to Train Checklist

Before starting training, verify:

### Environment
- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip list`)
- [ ] `torch`, `timm`, `mlflow` available

### GPU
- [ ] `nvidia-smi` shows GPU
- [ ] PyTorch detects CUDA: `torch.cuda.is_available()` returns `True`
- [ ] Can load CrossViT model without errors

### Data
- [ ] `data/processed/` folder exists
- [ ] CSV files present (train, val, test)
- [ ] `clahe_enhanced/` has 21,165 images
- [ ] Can load DataFrame without errors

### Notebooks
- [ ] All 6 training notebooks present
- [ ] Can open notebooks in Jupyter
- [ ] First few cells run without errors

### Directories
- [ ] `models/` directory exists (or will be created by notebooks)
- [ ] `results/` directory exists (or will be created)

---

## 🚀 Start Training

### Option 1: Run in Jupyter (Interactive)

```bash
cd notebooks
jupyter notebook  # or jupyter lab

# Open 06_crossvit_training.ipynb
# Run All Cells (Cell > Run All)
```

**Monitor in real-time**, can see progress bars

### Option 2: Convert to Python Script (Background)

```bash
cd notebooks

# Convert notebook to Python script
jupyter nbconvert --to script 06_crossvit_training.ipynb

# Run in background with logging
nohup python 06_crossvit_training.py > crossvit_training.log 2>&1 &

# Monitor progress
tail -f crossvit_training.log

# Check if still running
ps aux | grep crossvit_training
```

**Advantage:** Can close terminal, training continues

### Option 3: Use Screen/Tmux (Persistent Session)

```bash
# Start screen session
screen -S crossvit_training

# Activate venv
source venv/bin/activate

# Start training
cd notebooks
python 06_crossvit_training.py

# Detach: Ctrl+A, then D
# Reattach later: screen -r crossvit_training
```

**Advantage:** Can reconnect to session anytime

---

## 📊 Monitor Training Progress

### Check MLflow UI
```bash
cd notebooks
mlflow ui

# Open browser: http://localhost:5000
```

### Check GPU Usage
```bash
# Watch GPU in real-time
watch -n 2 nvidia-smi

# Or one-time check
nvidia-smi
```

### Check Log Files
```bash
tail -f crossvit_training.log
grep "Seed.*complete" crossvit_training.log
```

### Check Output Files
```bash
# Check model checkpoints
ls -lh ../models/

# Check results
ls -lh ../results/
```

---

## ⚡ Optimization Tips

### If Workstation Has More VRAM

Open notebooks, find CONFIG cell, increase batch sizes:

```python
# Example for CrossViT (06):
CONFIG = {
    'batch_size': 32,  # Instead of 8
    'gradient_accumulation_steps': 1,  # Disable accumulation
    # ... rest stays same
}
```

### Use All CPU Cores

```python
# In CONFIG:
'num_workers': 8,  # Instead of 0 (use CPU core count)
'pin_memory': True,
'persistent_workers': True,  # If using num_workers > 0
```

### Monitor Training Efficiency

```python
# Add to training loop (optional)
import time
batch_start = time.time()
# ... training code ...
batch_time = time.time() - batch_start
print(f"Batch time: {batch_time:.3f}s")
```

---

## 🔧 Troubleshooting

### Issue: "Module not found"
```bash
pip install <missing-module>
```

### Issue: "CUDA out of memory"
```python
# Reduce batch size in CONFIG cell
'batch_size': 4,  # Reduce further
```

### Issue: "File not found" for images
```python
# Check paths are correct
import os
print(os.path.abspath("../data/processed"))

# Verify image paths in CSV
df = pd.read_csv("../data/processed/train_processed.csv")
print(df['processed_path'].iloc[0])
print(os.path.exists(df['processed_path'].iloc[0]))
```

### Issue: MLflow errors
```bash
# Reinstall MLflow
pip uninstall mlflow -y
pip install mlflow
```

---

## 📝 Quick Reference Commands

### Transfer (Git Method)
```bash
# Laptop
git add .
git commit -m "Phase 2 notebooks complete"
git push

# Workstation
git clone https://github.com/YOUR_USERNAME/fyp-project.git
# + Copy data folder separately
```

### Setup on Workstation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install timm mlflow
```

### Test Setup
```bash
python -c "import torch, timm, mlflow; print('All imports OK')"
python -c "import torch; print(torch.cuda.is_available())"
```

### Start Training
```bash
cd notebooks
jupyter notebook  # Interactive
# or
python 06_crossvit_training.py  # Background
```

### Monitor
```bash
nvidia-smi
tail -f crossvit_training.log
cd notebooks && mlflow ui
```

---

## ✅ Expected Timeline

### Transfer
- USB: 15-30 minutes
- Network: 5-15 minutes
- Git + Data: 10-20 minutes
- Cloud: 30-60 minutes

### Setup on Workstation
- Environment: 5-10 minutes
- Verification: 5 minutes

### Training (Workstation Estimates)
- CrossViT: ~5 hours
- EfficientNet: ~2-3 hours
- DenseNet: ~2-3 hours
- Swin: ~4-5 hours
- ResNet-50: ~2-3 hours (skip if already done)
- ViT: ~6-8 hours

**Total: ~20-30 hours** (can run overnight/over weekend)

---

## 🎓 Best Practices

1. **Start with one model** (CrossViT) to verify setup
2. **Monitor first few epochs** to catch errors early
3. **Check GPU temperature** doesn't exceed 85°C
4. **Keep laptop backup** until workstation runs complete
5. **Use screen/tmux** for long-running jobs
6. **Verify results** after each model completes

---

**Transfer Method Recommendation:**

If comfortable with Git: **Method 1 (Git + USB for data)** ⭐
If want simplicity: **Method 2 (USB complete transfer)**

Both work perfectly!

---

**Created:** 2025-11-12
**For:** TAR UMT FYP Phase 2 Training
**Student:** Tan Ming Kai (24PMR12003)
