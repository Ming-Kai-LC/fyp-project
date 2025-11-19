# How to Check Training Status (PowerShell Guide)

**⚠️ IMPORTANT: This guide uses Windows PowerShell commands only!**
**Do NOT use Linux commands like `tail`, `grep`, `ls` - they won't work!**

**Generated:** 2025-11-18 16:00
**Training Started:** 2025-11-18 14:20
**Background Process ID:** 3eb855

---

## Common Linux → PowerShell Translations

**Don't Use Linux Commands!** Use these PowerShell equivalents instead:

| ❌ Linux Command | ✅ PowerShell Equivalent |
|------------------|--------------------------|
| `tail -f file.log` | `Get-Content file.log -Wait -Tail 20` |
| `tail -n 30 file.log` | `Get-Content file.log -Tail 30` |
| `grep "pattern" file` | `Select-String -Path file -Pattern "pattern"` |
| `ls -lh folder/` | `Get-ChildItem folder\ \| Format-Table` |
| `cat file` | `Get-Content file` |
| `ps aux \| grep python` | `Get-Process \| Where-Object {$_.ProcessName -eq "python"}` |

### Why Did `tail -f` Fail?

If you saw this error:
```
tail : The term 'tail' is not recognized as the name of a cmdlet, function, script file, or operable program.
```

**Reason:** `tail` is a Linux/Unix command. It doesn't exist in Windows PowerShell!

**Solution:** Use the PowerShell equivalent:
```powershell
Get-Content logs\resnet50_live_training.log -Wait -Tail 20
```

---

## Quick Status Check (Copy-Paste These Commands)

### 1. Check if Training is Still Running

```powershell
# Check if Python training process is active
Get-Process | Where-Object {$_.ProcessName -eq "python"} | Select-Object Id, ProcessName, CPU, WorkingSet

# If you see a process, training is still running
# If nothing appears, training has finished
```

### 2. View Latest Training Progress

```powershell
# View last 30 lines of training log
Get-Content logs\resnet50_live_training.log -Tail 30

# View only completed seeds
Select-String -Path logs\resnet50_live_training.log -Pattern "Seed.*complete"

# View only epoch summaries (recent 20)
Select-String -Path logs\resnet50_live_training.log -Pattern "Epoch [0-9]+:.*Val Acc" | Select-Object -Last 20
```

### 3. Check GPU Status

```powershell
# View GPU utilization and temperature
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader

# Full GPU status
nvidia-smi
```

### 4. Check Saved Model Files

```powershell
# List all saved models with size and timestamp
Get-ChildItem experiments\phase2_systematic\models\resnet50\ | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize

# Count how many seeds completed
(Get-ChildItem experiments\phase2_systematic\models\resnet50\*.pth).Count
```

### 5. Check Final Results (After Training Completes)

```powershell
# View final test accuracy for all seeds
Select-String -Path logs\resnet50_live_training.log -Pattern "Seed.*complete: Test Acc"

# View overall summary statistics
Select-String -Path logs\resnet50_live_training.log -Pattern "RESNET50 Results"
```

---

## Step-by-Step Guide for PowerShell

### Opening PowerShell

1. Press `Win + X`
2. Select "Windows PowerShell" or "Terminal"
3. Navigate to project directory:
   ```powershell
   cd C:\Users\FOCS3\Documents\GitHub\fyp-project\FYP_Code
   ```

### Understanding the Output

#### Training is Running If:
- `Get-Process` shows a Python process with high CPU usage
- GPU utilization is 95-99%
- Log file shows recent timestamps (within last 5 minutes)

#### Training is Complete If:
- No Python training process found
- You see "Seed 101112 complete" in logs
- You see "RESNET50 Results (5 seeds):" summary in logs
- 5 model files exist in `experiments\phase2_systematic\models\resnet50\`

---

## Detailed Status Commands

### Check Training Progress

```powershell
# View training progress with epoch details
Get-Content logs\resnet50_live_training.log -Tail 50 | Select-String "Epoch [0-9]+:|Seed.*complete|Best model"
```

**What to Look For:**
- Current epoch number (e.g., "Epoch 15: Train Loss=0.0320 | Val Acc=95.18%")
- Best model saved messages
- Seed completion messages

### Monitor Training in Real-Time

```powershell
# Watch log file continuously (like tail -f)
Get-Content logs\resnet50_live_training.log -Wait -Tail 20
```

**To Stop Watching:** Press `Ctrl + C`

### Check Expected Completion Time

**Current Status (as of 16:00):**
- Seed 42: ✅ Complete (95.28%)
- Seed 123: ✅ Complete (95.98%)
- Seed 456: 🔄 In progress (Epoch ~17/30)
- Seed 789: ⏸️ Pending
- Seed 101112: ⏸️ Pending

**Time Estimates:**
- Seed 456: ~10-15 minutes remaining
- Seed 789: ~25 minutes
- Seed 101112: ~25 minutes
- **Total remaining:** ~60-65 minutes from 16:00 (Complete by ~17:00-17:10)

---

## Checking for Errors

### Look for Training Errors

```powershell
# Search for error messages
Select-String -Path logs\resnet50_live_training.log -Pattern "error|Error|ERROR|exception|Exception|Traceback" | Select-Object -Last 10
```

### Check for GPU Out of Memory Errors

```powershell
# Search for CUDA/GPU errors
Select-String -Path logs\resnet50_live_training.log -Pattern "CUDA|out of memory|OOM" | Select-Object -Last 5
```

**If Errors Found:**
- Training may have stopped early
- Check GPU status with `nvidia-smi`
- Restart training with: `python train_all_models_safe.py resnet50`

---

## After Training Completes

### 1. Verify All Seeds Completed

```powershell
# Should show 5 completion messages
Select-String -Path logs\resnet50_live_training.log -Pattern "Seed.*complete: Test Acc"
```

**Expected Output:**
```
Seed 42 complete: Test Acc = 95.28%
Seed 123 complete: Test Acc = 95.98%
Seed 456 complete: Test Acc = XX.XX%
Seed 789 complete: Test Acc = XX.XX%
Seed 101112 complete: Test Acc = XX.XX%
```

### 2. View Summary Statistics

```powershell
# View mean and standard deviation across 5 seeds
Select-String -Path logs\resnet50_live_training.log -Pattern "RESNET50 Results|Mean" | Select-Object -Last 5
```

**Expected Output:**
```
RESNET50 Results (5 seeds):
Mean +/- Std: XX.XX% +/- X.XX%
```

### 3. Check All Model Files Saved

```powershell
# Should show 5 .pth files
Get-ChildItem experiments\phase2_systematic\models\resnet50\*.pth | Select-Object Name, @{Name="Size (MB)";Expression={[math]::Round($_.Length/1MB,2)}}, LastWriteTime
```

**Expected Output:**
```
Name                          Size (MB) LastWriteTime
----                          --------- -------------
resnet50_best_seed42.pth      91.00     2025-11-18 14:31
resnet50_best_seed123.pth     91.00     2025-11-18 15:14
resnet50_best_seed456.pth     91.00     2025-11-18 15:XX
resnet50_best_seed789.pth     91.00     2025-11-18 16:XX
resnet50_best_seed101112.pth  91.00     2025-11-18 16:XX
```

### 4. View Confusion Matrices (If Generated)

```powershell
# Check for confusion matrix images
Get-ChildItem experiments\phase2_systematic\results\confusion_matrices\resnet50*.png | Select-Object Name, LastWriteTime
```

### 5. View Results CSV

```powershell
# Display results table
Import-Csv experiments\phase2_systematic\results\metrics\resnet50_results.csv | Format-Table -AutoSize
```

---

## Next Steps After ResNet-50 Completes

### Option 1: Train Next Model (DenseNet-121)

```powershell
# Start DenseNet-121 training (estimated 3 hours)
python train_all_models_safe.py densenet121
```

### Option 2: Train All Remaining Models

```powershell
# Train all 5 remaining models automatically (estimated 15-18 hours)
python train_all_models_safe.py all
```

**Recommended:** Train one model at a time for easier monitoring and debugging.

### Training Order (Recommended):
1. ✅ ResNet-50 (~2.5 hours) - IN PROGRESS
2. DenseNet-121 (~3 hours)
3. EfficientNet-B0 (~2.5 hours)
4. ViT-Base (~4 hours)
5. Swin-Tiny (~3 hours)
6. CrossViT-Tiny (~3 hours)

**Total:** ~18-20 hours for all 6 models

---

## Viewing MLflow Results

### Start MLflow UI

```powershell
# Start MLflow dashboard (keep terminal open)
mlflow ui --backend-store-uri file:.\experiments\phase2_systematic\mlruns

# Then open in browser: http://localhost:5000
```

**What You'll See:**
- All 30 training runs (6 models × 5 seeds)
- Sortable by accuracy, loss, etc.
- All hyperparameters logged
- All confusion matrices attached

**To Stop MLflow:** Press `Ctrl + C` in the terminal

---

## Troubleshooting

### Problem: Training Process Not Found But Not Complete

```powershell
# Check if process crashed
Select-String -Path logs\resnet50_live_training.log -Pattern "error|Error|exception|Traceback" -Context 0,5 | Select-Object -Last 3
```

**Solution:** Restart training
```powershell
python train_all_models_safe.py resnet50
```
Training will skip already completed seeds automatically.

### Problem: GPU Not Utilized

```powershell
# Check GPU status
nvidia-smi

# Check if other users are using GPU
nvidia-smi pmon -c 1
```

### Problem: Training Too Slow

**Current Status:** Training at 98-99% GPU utilization is OPTIMAL.

**If slower (<50% GPU):**
- Other users may be using the workstation
- Check with: `nvidia-smi pmon`

---

## Quick Reference Card

| Task | Command |
|------|---------|
| **Check if running** | `Get-Process | Where-Object {$_.ProcessName -eq "python"}` |
| **View last 30 lines** | `Get-Content logs\resnet50_live_training.log -Tail 30` |
| **Watch real-time** | `Get-Content logs\resnet50_live_training.log -Wait -Tail 20` |
| **GPU status** | `nvidia-smi` |
| **Completed seeds** | `Select-String -Path logs\resnet50_live_training.log -Pattern "Seed.*complete"` |
| **Model files** | `Get-ChildItem experiments\phase2_systematic\models\resnet50\` |
| **Check errors** | `Select-String -Path logs\resnet50_live_training.log -Pattern "error\|Error\|exception" \| Select-Object -Last 10` |
| **Start next model** | `python train_all_models_safe.py densenet121` |
| **MLflow dashboard** | `mlflow ui --backend-store-uri file:.\experiments\phase2_systematic\mlruns` |

---

## Expected Timeline

**Started:** 2025-11-18 14:20
**Current Time:** ~16:00
**Expected Completion:** ~17:00-17:10 (about 1 hour remaining)

**What's Done:**
- ✅ Seed 42 complete (95.28%)
- ✅ Seed 123 complete (95.98%)

**What's Remaining:**
- 🔄 Seed 456 (in progress, ~70% done)
- ⏸️ Seed 789 (~25 minutes)
- ⏸️ Seed 101112 (~25 minutes)

---

## Contact Information

**Project:** TAR UMT FYP - CrossViT COVID-19 Classification
**Student:** Tan Ming Kai (24PMR12003)
**Workstation:** FOCS3 (RTX 6000 Ada, 51.5 GB VRAM)

**Status File:** `TRAINING_STATUS.md` (snapshot when you left)
**This File:** `CHECK_TRAINING_STATUS.md` (how to check now)

---

**Last Updated:** 2025-11-18 16:00
**Training Status:** ✅ Running smoothly at 99% GPU utilization
