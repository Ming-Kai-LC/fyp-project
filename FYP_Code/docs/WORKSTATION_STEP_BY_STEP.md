# Workstation Training - Step-by-Step Checklist

**Student:** Tan Ming Kai (24PMR12003)
**Date:** 2025-11-13
**Project:** CrossViT COVID-19 Classification

---

## BEFORE YOU GO TO LAB - PREPARATION

- [ ] Read this document completely
- [ ] Ensure laptop is charged
- [ ] Know your GitHub credentials (if needed)
- [ ] Project location: `D:\Users\USER\Documents\GitHub\fyp-project\FYP_Code`

---

## PART 1: CONNECT TO WORKSTATION (5 minutes)

### Step 1.1: Connect to Wi-Fi
- [ ] Connect to **AINexus24** Wi-Fi
- [ ] Password: `12345678`
- [ ] Wait for connection to establish

### Step 1.2: Open Remote Desktop
- [ ] Press `Win + R` on your laptop
- [ ] Type: `mstsc`
- [ ] Press Enter

### Step 1.3: Connect to Workstation
- [ ] In "Computer" field, type: `metacode1` (or `metacode2`)
- [ ] Click "Connect"
- [ ] If asked to trust, click "Yes"

### Step 1.4: Login
- [ ] Username: `focs1` (or `focs2` or `focs3` - use assigned account)
- [ ] Password: `focs123`
- [ ] Press Enter

**You should now see the workstation desktop!**

---

## PART 2: TRANSFER PROJECT FILES (10-15 minutes)

### Step 2.1: Open File Explorer on Workstation
- [ ] Click File Explorer icon on taskbar
- [ ] You should see your laptop drives as `\\tsclient\C`, `\\tsclient\D`, etc.

### Step 2.2: Navigate to Your Project
- [ ] In address bar, type or navigate to:
  ```
  \\tsclient\D\Users\USER\Documents\GitHub\fyp-project\FYP_Code
  ```
- [ ] You should see the FYP_Code folder from your laptop

### Step 2.3: Copy to Workstation
- [ ] Right-click on `FYP_Code` folder
- [ ] Click "Copy"
- [ ] Navigate to: `C:\Users\focs1\` (or focs2/focs3)
- [ ] Right-click in empty space
- [ ] Click "Paste"
- [ ] **Wait for transfer to complete** (8-15 minutes, 2.4 GB)

**IMPORTANT:** Do NOT close remote desktop while copying!

### Step 2.4: Verify Transfer Complete
- [ ] Check that copying is finished (progress bar disappears)
- [ ] Open `C:\Users\focs1\FYP_Code`
- [ ] Verify folders exist: `data`, `models`, `notebooks`, `results`, `src`

---

## PART 3: SETUP ENVIRONMENT (10-15 minutes)

### Step 3.1: Open Command Prompt or PowerShell
- [ ] Press `Win + R`
- [ ] Type: `cmd` or `powershell`
- [ ] Press Enter

### Step 3.2: Navigate to Project
```bash
cd C:\Users\focs1\FYP_Code
```
- [ ] Type the command above and press Enter
- [ ] Verify you're in correct folder: `dir` should show data, models, notebooks, etc.

### Step 3.3: Check Python Installation
```bash
python --version
```
- [ ] Should show Python 3.8 or higher
- [ ] If not installed, ask lab staff for help

### Step 3.4: Create Virtual Environment
```bash
python -m venv venv
```
- [ ] Wait 1-2 minutes for creation
- [ ] You should see a new `venv` folder

### Step 3.5: Activate Virtual Environment

**Windows Command Prompt:**
```bash
venv\Scripts\activate
```

**Windows PowerShell:**
```bash
venv\Scripts\Activate.ps1
```

- [ ] Command prompt should now show `(venv)` at the beginning
- [ ] If you get an error about execution policy (PowerShell only), run:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### Step 3.6: Install Dependencies
```bash
pip install -r requirements.txt
```
- [ ] Wait 5-10 minutes for installation
- [ ] Should see "Successfully installed..." messages

### Step 3.7: Install Additional Packages
```bash
pip install timm mlflow
```
- [ ] Wait 1-2 minutes
- [ ] These are required for training

---

## PART 4: VERIFY SETUP (5 minutes)

### Step 4.1: Check GPU Availability
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```
- [ ] Should print: `CUDA Available: True`
- [ ] If False, GPU is not detected - ask lab staff

### Step 4.2: Check GPU Name
```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```
- [ ] Should show workstation GPU name (e.g., "NVIDIA RTX 4090", "RTX 3090", etc.)
- [ ] Note down GPU name: ________________

### Step 4.3: Verify Data Integrity
```bash
python -c "import os; count = sum([len([f for f in files if f.endswith('.png')]) for root, dirs, files in os.walk('data/processed/clahe_enhanced')]); print(f'Total images: {count}')"
```
- [ ] Should print: `Total images: 21165`
- [ ] If different number, data transfer may be incomplete

### Step 4.4: Check Model Files
```bash
dir models
```
- [ ] Should see 5 ResNet-50 checkpoint files (.pth)
- [ ] Total ~451 MB

**If all checks pass, you're ready to train!** ✅

---

## PART 5: START TRAINING (5-10 minutes)

### Option A: Using Jupyter Notebook (Recommended - Easy to Monitor)

#### Step 5A.1: Install Jupyter (if not already installed)
```bash
pip install jupyter
```

#### Step 5A.2: Start Jupyter
```bash
cd notebooks
jupyter notebook
```
- [ ] Browser should open automatically
- [ ] If not, copy the URL shown (e.g., `http://localhost:8888/?token=...`)

#### Step 5A.3: Open Training Notebook
- [ ] In browser, click on: `06_crossvit_training.ipynb`
- [ ] Notebook should open

#### Step 5A.4: Run All Cells
- [ ] Click "Kernel" → "Restart & Run All"
- [ ] Confirm restart
- [ ] Training will start automatically

#### Step 5A.5: Monitor Progress
- [ ] You should see progress bars for each epoch
- [ ] First seed will take ~1-1.5 hours
- [ ] Total for all 5 seeds: ~6-8 hours

**You can now leave the lab and let it train overnight!**

---

### Option B: Using Python Script (Background - No Monitoring)

#### Step 5B.1: Navigate to Notebooks
```bash
cd notebooks
```

#### Step 5B.2: Run Training Script
```bash
python -c "exec(open('06_crossvit_training.ipynb').read())"
```

**OR convert notebook to Python script first:**
```bash
jupyter nbconvert --to script 06_crossvit_training.ipynb
python 06_crossvit_training.py > crossvit_training.log 2>&1
```

**This runs in background. Check log file later to see progress.**

---

## PART 6: USING CLAUDE CODE ON WORKSTATION (HIGHLY RECOMMENDED!)

### Why Use Claude Code on Workstation?

Claude Code will help you:
- ✅ Fix any errors that occur during training
- ✅ Monitor training progress
- ✅ Adjust configurations if needed
- ✅ Debug GPU/CUDA issues
- ✅ Optimize batch sizes for better performance
- ✅ Answer questions without needing to search online

### Step 6.1: Install Claude Code on Workstation

**Option 1: Download from Website**
- [ ] Open browser: `https://claude.com/code`
- [ ] Click "Download"
- [ ] Install Claude Code for Windows
- [ ] Login with your Anthropic account

**Option 2: If Already Installed on Workstation**
- [ ] Check if Claude Code is already installed
- [ ] Open it and login

### Step 6.2: Open Project in Claude Code
- [ ] Open Claude Code
- [ ] Click "Open Folder"
- [ ] Navigate to: `C:\Users\focs1\FYP_Code`
- [ ] Select the folder

### Step 6.3: Claude Code Will Read CLAUDE.md Automatically!

**IMPORTANT:** Your project has a `CLAUDE.md` file that contains:
- Project context (CrossViT COVID-19 classification)
- Hardware constraints (GPU VRAM limits)
- All model specifications
- Training configurations
- Dataset details
- Common pitfalls to avoid

**Claude Code will automatically read this file and understand your project!**

### Step 6.4: How to Use Claude Code During Training

**Example 1: If You Get an Error**
```
You: I got this error: "CUDA out of memory"
Claude Code: [Reads CLAUDE.md, knows your constraints]
             Reduce batch_size in notebook CONFIG from 8 to 4.
             I'll help you edit the notebook...
```

**Example 2: Optimize for Workstation GPU**
```
You: The workstation has RTX 4090 with 24GB VRAM. Can I speed up training?
Claude Code: [Checks CLAUDE.md, sees current config is for 8GB]
             Yes! You can increase batch_size to 32 and disable gradient
             accumulation. This will be 3x faster. Let me update the config...
```

**Example 3: Monitor Progress**
```
You: Check training progress for CrossViT
Claude Code: [Reads MLflow data or log files]
             Seed 42: Epoch 15/50, Accuracy: 92.3%
             Estimated time remaining: 45 minutes
```

**Example 4: Start Next Model**
```
You: CrossViT training finished. What should I do next?
Claude Code: [Reads CLAUDE.md Phase 2 sequence]
             Great! Next is DenseNet-121 (08_densenet121_training.ipynb).
             Let me help you start it...
```

### Step 6.5: Useful Claude Code Commands

In Claude Code chat, you can say:
- `"Check if GPU is available"`
- `"Show me training progress"`
- `"What notebooks should I run next?"`
- `"Fix this error: [paste error message]"`
- `"How much VRAM is being used?"`
- `"Can I increase batch size on this GPU?"`
- `"Monitor CrossViT training"`
- `"Read the training log"`

**Claude Code will help you like I'm helping you now, but on the workstation!**

---

## PART 7: LEAVING THE LAB (After Starting Training)

### Step 7.1: Verify Training is Running
- [ ] Check that progress bars are moving (Jupyter)
- [ ] OR check log file is being written (Python script)
- [ ] Note the current epoch/seed

### Step 7.2: Note Important Info
Write down:
- [ ] Workstation used: metacode1 / metacode2
- [ ] Account used: focs1 / focs2 / focs3
- [ ] Training started: _____ (time)
- [ ] Current model: 06_crossvit_training
- [ ] Expected completion: _____ (time)

### Step 7.3: Keep Remote Desktop Open (Optional)
- [ ] You can disconnect and training will continue
- [ ] Minimize remote desktop window
- [ ] Training runs on workstation, not your laptop

### Step 7.4: You Can Close Everything
- [ ] Training continues even if you disconnect
- [ ] Close remote desktop if you want
- [ ] Training runs independently on workstation

---

## PART 8: NEXT DAY - COLLECT RESULTS & CLEANUP

### Step 8.1: Reconnect to Workstation
- [ ] Connect to AINexus24 Wi-Fi
- [ ] Remote desktop to same workstation (metacode1/metacode2)
- [ ] Login with same account (focs1/focs2/focs3)

### Step 8.2: Check Training Status

**If using Jupyter:**
- [ ] Open browser to Jupyter (might need to restart)
- [ ] Check notebook - should show "Kernel: Idle" if finished

**If using Python script:**
```bash
cd C:\Users\focs1\FYP_Code\notebooks
tail crossvit_training.log
```
- [ ] Check log shows completion message

### Step 8.3: Verify Results Created
```bash
cd C:\Users\focs1\FYP_Code
dir models
dir results
```

**Should see new files:**
- [ ] `models/crossvit_best_seed42.pth`
- [ ] `models/crossvit_best_seed123.pth`
- [ ] `models/crossvit_best_seed456.pth`
- [ ] `models/crossvit_best_seed789.pth`
- [ ] `models/crossvit_best_seed101112.pth`
- [ ] `results/crossvit_*.png` (confusion matrices)
- [ ] `results/crossvit_results.csv`

### Step 8.4: Copy Results Back to Laptop

**In Remote Desktop:**
- [ ] Open File Explorer
- [ ] Navigate to: `C:\Users\focs1\FYP_Code\models`
- [ ] Select CrossViT checkpoint files
- [ ] Copy them
- [ ] Navigate to: `\\tsclient\D\Users\USER\Documents\GitHub\fyp-project\FYP_Code\models`
- [ ] Paste files
- [ ] Repeat for `results` folder

**OR copy entire updated project:**
- [ ] Copy entire `C:\Users\focs1\FYP_Code` folder
- [ ] Paste to `\\tsclient\D\Users\USER\Documents\GitHub\fyp-project\FYP_Code_workstation`

### Step 8.5: Start Next Model (Optional - If Time Permits)

**If you have more time booked:**
- [ ] Open `08_densenet121_training.ipynb` in Jupyter
- [ ] Click "Restart & Run All"
- [ ] Let it train overnight again

**Training sequence (from CLAUDE.md):**
1. ✅ ResNet-50 (already done - 95.49%)
2. ⏭️ CrossViT (just completed)
3. ⏭️ DenseNet-121 (08) - Next
4. ⏭️ EfficientNet-B0 (09)
5. ⏭️ ViT-Base/16 (10)
6. ⏭️ Swin-Tiny (11)

### Step 8.6: CLEANUP WORKSTATION (MANDATORY!)

**IMPORTANT:** Lab rules require you to delete ALL data!

```bash
cd C:\Users\focs1
rmdir /s /q FYP_Code
```

**OR manually:**
- [ ] Delete entire `C:\Users\focs1\FYP_Code` folder
- [ ] Empty Recycle Bin
- [ ] Clear browser history/downloads

**Verify cleanup:**
- [ ] Check `C:\Users\focs1` - should NOT see FYP_Code
- [ ] You've copied results back to laptop already

**If you don't clean up:** You'll be banned from using the lab (per lab rules)

### Step 8.7: Logout
- [ ] Close all applications
- [ ] Click Start → Logout
- [ ] Close remote desktop on your laptop

---

## QUICK REFERENCE - TRAINING TIMELINE

### First Day (1:30 PM - 3:00 PM):
- 1:30-1:35 PM: Connect to workstation
- 1:35-1:50 PM: Transfer files (15 min)
- 1:50-2:05 PM: Setup environment (15 min)
- 2:05-2:10 PM: Verify GPU/data (5 min)
- 2:10-2:20 PM: Start training (10 min)
- 2:20 PM: **Leave lab, training runs overnight**

### Next Day (1:30 PM - 3:00 PM):
- 1:30-1:35 PM: Reconnect to workstation
- 1:35-1:40 PM: Check results (5 min)
- 1:40-2:00 PM: Copy results to laptop (20 min)
- 2:00-2:10 PM: Start next model OR cleanup (10 min)
- 2:10-2:15 PM: Final cleanup if done (5 min)
- 2:15 PM: **Logout, all done!**

---

## TROUBLESHOOTING GUIDE

### Problem: GPU not detected (CUDA Available: False)
**Solution:**
1. Check if NVIDIA drivers installed: `nvidia-smi`
2. If not, ask lab staff
3. Or use Claude Code: "GPU not detected, help me fix it"

### Problem: Out of memory error during training
**Solution:**
1. Reduce `batch_size` in notebook CONFIG
2. Change from 8 to 4 or even 2
3. Or ask Claude Code: "Got CUDA out of memory error, help me fix it"

### Problem: Transfer is very slow (>30 minutes)
**Solution:**
1. Check Wi-Fi signal strength
2. Try pausing and resuming
3. Or compress folder first (zip), then transfer

### Problem: Python not found
**Solution:**
1. Ask lab staff to help install Python
2. Or check if it's in different location
3. Try: `python3 --version` or `py --version`

### Problem: Can't activate virtual environment (PowerShell error)
**Solution:**
1. Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
2. Try activating again
3. Or use Command Prompt instead of PowerShell

### Problem: Training seems stuck
**Solution:**
1. Check GPU utilization: `nvidia-smi`
2. Should show Python using GPU
3. Check log files for errors
4. Or ask Claude Code: "Is training stuck? Check progress"

### Problem: Jupyter won't start
**Solution:**
1. Make sure virtual environment is activated
2. Try: `pip install jupyter` again
3. Or run as Python script instead (Option B)

---

## FINAL CHECKLIST - BEFORE YOU LEAVE

- [ ] Training is running (verified progress bars moving)
- [ ] GPU is being used (check nvidia-smi)
- [ ] Noted which workstation/account you're using
- [ ] Noted expected completion time
- [ ] Have Claude Code installed on workstation (optional but helpful)
- [ ] Know when you'll return to collect results

**You're all set! Training will run overnight automatically.** 🌙

---

## NOTES SECTION (Write Your Own Notes Here)

**Workstation Used:** ________________

**Account Used:** ________________

**Date Started:** ________________

**Time Started:** ________________

**Model Training:** ________________

**Expected Finish:** ________________

**GPU Name:** ________________

**Any Issues:**
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________

**Results Location:**
_________________________________________________________________

---

## HELP & SUPPORT

**During Training:**
- Use Claude Code on workstation (it has all context from CLAUDE.md)
- Ask Claude Code to help with any errors
- Check WORKSTATION_TRANSFER_GUIDE.md for detailed info

**Documentation Files in Project:**
- `CLAUDE.md` - Complete project context
- `WORKSTATION_TRANSFER_GUIDE.md` - Detailed transfer guide
- `NOTEBOOK_VALIDATION_REPORT.md` - Notebook validation
- `REMOTE_DESKTOP_TRANSFER_READY.md` - What was cleaned up
- `PHASE2_SETUP.md` - Phase 2 overview

**Emergency Contact:**
- Lab staff for hardware/network issues
- Your supervisor for project questions

---

**Good luck with your training!** 🚀

Remember: Training can run overnight. You don't need to babysit it!
