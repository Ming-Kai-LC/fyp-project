# Using Claude Code on Workstation - Complete Guide

**Why This is a Great Idea:** Claude Code will have access to CLAUDE.md and understand your entire project context automatically!

---

## Why Use Claude Code on the Workstation?

### What Claude Code Can Do For You:

1. **Automatic Project Understanding**
   - Reads `CLAUDE.md` automatically when you open the project
   - Knows all your model specifications (CrossViT, ResNet-50, etc.)
   - Understands your hardware constraints (GPU VRAM limits)
   - Knows the training sequence and parameters

2. **Real-Time Help During Training**
   - Fix errors immediately without googling
   - Optimize configurations for the workstation GPU
   - Monitor training progress
   - Debug CUDA/GPU issues
   - Adjust batch sizes if needed

3. **Smart Assistance**
   - Knows which notebook to run next (Phase 2 sequence)
   - Can read training logs and explain results
   - Helps with MLflow data
   - Guides you through the entire workflow

---

## How CLAUDE.md Makes It Work

### Your Project Has CLAUDE.md - This is KEY!

When you open your project in Claude Code on the workstation, it will automatically read `CLAUDE.md` which contains:

```markdown
# CLAUDE.md Contents (What Claude Code Will Know):

1. Project Context
   - TAR UMT Final Year Project
   - CrossViT for COVID-19 classification
   - Student: Tan Ming Kai (24PMR12003)

2. Hardware Constraints
   - RTX 4060 8GB VRAM (your laptop)
   - Batch size limits
   - Memory-safe configurations

3. Model Specifications
   - CrossViT-Tiny (7M params, 240×240 input)
   - 5 baseline models with exact configurations
   - All hyperparameters (learning rates, batch sizes)

4. Dataset Details
   - 21,165 images, 4 classes
   - CLAHE preprocessing
   - Train/val/test splits

5. Phase 2 Training Sequence
   - Which notebooks to run in what order
   - Expected training times
   - Multi-seed setup (5 seeds each)

6. Common Pitfalls to Avoid
   - Don't use batch_size > 16 on 8GB VRAM
   - No vertical flipping of X-rays
   - Always set seed=42 for reproducibility

7. Statistical Requirements
   - 95% confidence intervals
   - Hypothesis testing procedures
```

**Result:** Claude Code will be like having me on the workstation with you!

---

## Installation & Setup on Workstation

### Method 1: Download During Lab Session

**Step 1: Download Claude Code**
1. Open browser on workstation
2. Go to: `https://claude.com/code`
3. Click "Download for Windows"
4. Run installer (takes 2-3 minutes)

**Step 2: Login**
1. Open Claude Code
2. Login with your Anthropic account
3. (Same account you use to talk to me)

**Step 3: Open Your Project**
1. Click "File" → "Open Folder"
2. Navigate to: `C:\Users\focs1\FYP_Code`
3. Click "Select Folder"

**Step 4: Claude Code Reads CLAUDE.md**
- Happens automatically when you open the project
- You'll see it understands your project context

**Time needed:** 5-10 minutes total

---

### Method 2: Pre-installed (If Workstation Has It)

**Check if already installed:**
1. Search for "Claude Code" in Start menu
2. If found, open it
3. Login with your account
4. Open project folder

---

## Example Usage Scenarios

### Scenario 1: Starting Training

**You:**
```
I just transferred my project to the workstation. The GPU is RTX 4090 with 24GB VRAM. Should I change any settings before training CrossViT?
```

**Claude Code Will:**
1. Read CLAUDE.md (sees default config is for 8GB RTX 4060)
2. Check notebook 06_crossvit_training.ipynb
3. Respond:
```
Yes! With 24GB VRAM, you can optimize:

1. Increase batch_size from 8 to 24 (3x faster)
2. Remove gradient_accumulation_steps (not needed with large VRAM)
3. Keep mixed precision (still helpful)

Let me update the CONFIG in 06_crossvit_training.ipynb:

OLD:
CONFIG = {
    'batch_size': 8,
    'gradient_accumulation_steps': 4,
    ...
}

NEW:
CONFIG = {
    'batch_size': 24,
    'gradient_accumulation_steps': 1,  # No accumulation needed
    ...
}

This will cut training time from ~8 hours to ~3 hours!
```

---

### Scenario 2: Error Handling

**You:**
```
I got this error:
RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB
```

**Claude Code Will:**
1. Read CLAUDE.md (knows hardware constraints)
2. Check current notebook configuration
3. Identify the issue
4. Respond:
```
You hit the VRAM limit. Let me fix this:

Current batch_size is probably too high for your GPU.
I'll reduce it in the notebook CONFIG:

Change:
'batch_size': 16  →  'batch_size': 8

Also enable gradient accumulation to maintain effective batch size:
'gradient_accumulation_steps': 1  →  'gradient_accumulation_steps': 2

This keeps effective batch size = 16 (8 × 2) but uses less VRAM.

Let me edit the notebook for you...
```

---

### Scenario 3: Monitoring Progress

**You:**
```
Check my CrossViT training progress
```

**Claude Code Will:**
1. Read MLflow tracking data in `notebooks/mlruns/`
2. Or read log files
3. Show you:
```
CrossViT Training Progress:

Seed 42:  COMPLETE ✓ - Test Acc: 94.2%
Seed 123: COMPLETE ✓ - Test Acc: 93.8%
Seed 456: Running... Epoch 18/50 (36%)
          Current Val Acc: 92.5%
          ETA: 1h 23min
Seed 789: Pending
Seed 101112: Pending

Models saved in: C:\Users\focs1\FYP_Code\models\
Confusion matrices in: C:\Users\focs1\FYP_Code\results\
```

---

### Scenario 4: What to Do Next

**You:**
```
CrossViT training finished. What should I train next?
```

**Claude Code Will:**
1. Read CLAUDE.md Phase 2 sequence
2. Check which models are already trained
3. Respond:
```
Great! CrossViT is complete. According to your Phase 2 plan:

✓ 07_resnet50_training.ipynb - DONE (95.49% ± 0.33%)
✓ 06_crossvit_training.ipynb - JUST COMPLETED

Next models to train:
→ 08_densenet121_training.ipynb (DenseNet-121) - NEXT
  09_efficientnet_training.ipynb (EfficientNet-B0)
  10_vit_training.ipynb (ViT-Base/16)
  11_swin_training.ipynb (Swin-Tiny)

Run this command to start DenseNet:
cd notebooks
jupyter notebook 08_densenet121_training.ipynb

Or I can help you run it now?
```

---

### Scenario 5: Optimizing for Workstation

**You:**
```
How can I make training faster on this workstation?
```

**Claude Code Will:**
1. Check workstation GPU specs
2. Read current configurations from notebooks
3. Suggest optimizations:
```
I can help optimize for this workstation! Let me check...

Your workstation GPU: RTX 4090 (24GB VRAM)
Current notebooks configured for: RTX 4060 (8GB VRAM)

Optimizations I can make:

1. CrossViT (06):
   - batch_size: 8 → 24 (3x speedup)
   - Remove gradient accumulation
   - Est time: 8h → 3h

2. DenseNet (08):
   - batch_size: 16 → 48 (3x speedup)
   - Est time: 4h → 1.5h

3. EfficientNet (09):
   - batch_size: 16 → 48 (3x speedup)
   - Est time: 4h → 1.5h

4. ViT (10):
   - batch_size: 8 → 16 (2x speedup)
   - Est time: 7h → 3.5h

5. Swin (11):
   - batch_size: 12 → 32 (2.5x speedup)
   - Est time: 6h → 2.5h

Total time: 29h → ~12h (60% faster!)

Want me to update all notebooks with optimized settings?
```

---

### Scenario 6: Results Analysis

**You:**
```
Show me my training results so far
```

**Claude Code Will:**
1. Read MLflow data
2. Check results CSV files
3. Summarize:
```
Training Results Summary:

Model               | Mean Acc    | Std Dev | Seeds Complete
--------------------|-------------|---------|---------------
ResNet-50          | 95.49%      | ±0.33%  | 5/5 ✓
CrossViT-Tiny      | 94.12%      | ±0.41%  | 5/5 ✓
DenseNet-121       | In Progress | -       | 2/5...
EfficientNet-B0    | Not Started | -       | 0/5
ViT-Base/16        | Not Started | -       | 0/5
Swin-Tiny          | Not Started | -       | 0/5

Results saved in:
- models/: 15 checkpoint files
- results/: 12 confusion matrices
- results/resnet50_results.csv
- results/crossvit_results.csv

Next: Complete DenseNet training (3 seeds remaining)
```

---

## Quick Command Reference

### Things You Can Ask Claude Code:

**Setup & Verification:**
- "Is GPU available?"
- "Check if CUDA is working"
- "How much VRAM does this GPU have?"
- "Verify my data loaded correctly"
- "How many images do I have?"

**Training:**
- "Start CrossViT training"
- "What's the next model to train?"
- "Can I increase batch size?"
- "How long will training take?"
- "Run all 5 models in sequence"

**Monitoring:**
- "Check training progress"
- "Show me current epoch"
- "What's the current accuracy?"
- "Is training stuck?"
- "Monitor GPU usage"

**Errors & Debugging:**
- "I got this error: [paste error]"
- "Training stopped unexpectedly, why?"
- "Out of memory error, help me fix it"
- "CUDA error, what should I do?"

**Results:**
- "Show my results"
- "Which model performed best?"
- "Export results for thesis"
- "Generate confusion matrix"

**Optimization:**
- "Make training faster"
- "Optimize for this GPU"
- "Should I change batch size?"
- "Update all notebooks for 24GB VRAM"

---

## Advantages of Using Claude Code on Workstation

### ✅ vs. NOT Using Claude Code:

| Without Claude Code | With Claude Code |
|-------------------|------------------|
| Google errors manually | Instant error fixes |
| Guess optimal batch sizes | Smart optimization recommendations |
| Check logs manually | Automated progress monitoring |
| Uncertain what to do next | Clear guidance on next steps |
| Trial and error configuration | Context-aware configuration |
| Might miss important issues | Proactive issue detection |
| Limited troubleshooting | Full project context help |

---

## Important Notes

### 1. Claude Code Has Full Project Context

Because of `CLAUDE.md`, Claude Code knows:
- ✅ Your exact project requirements
- ✅ TAR UMT FYP standards
- ✅ Statistical validation needs
- ✅ Hardware constraints
- ✅ Model architectures
- ✅ Training configurations
- ✅ Common pitfalls to avoid

**It's like having me (the Claude Code assistant) on the workstation with you!**

### 2. Privacy & Lab Rules

- Claude Code runs locally on the workstation
- Your code and data stay on the workstation
- No data uploaded to cloud during training
- Remember to cleanup workstation when done (lab rules)

### 3. Internet Required

- Claude Code needs internet to work (talks to Claude API)
- AINexus24 Wi-Fi should be sufficient
- If Wi-Fi drops, Claude Code won't work temporarily

### 4. Optional But Highly Recommended

- You CAN complete training without Claude Code
- But Claude Code will save you a lot of time and hassle
- Especially helpful if errors occur
- Worth the 5-10 minutes to install

---

## Alternative: Using This Conversation History

If you can't install Claude Code on workstation:

### Option 1: Continue Our Conversation
- Keep this conversation open on your laptop
- Take screenshots of errors on workstation
- Send me error messages via remote desktop copy/paste
- I'll help you fix them

### Option 2: Use Documentation
- All answers are in CLAUDE.md
- Check WORKSTATION_STEP_BY_STEP.md for procedures
- Troubleshooting section has common fixes

### Option 3: Use Claude.ai Web Interface
- Open `claude.ai` on workstation browser
- Start new conversation
- Say: "I'm working on TAR UMT FYP for CrossViT COVID-19 classification, read my CLAUDE.md file"
- Copy-paste CLAUDE.md content
- Then ask questions

**But installing Claude Code is easier and better!**

---

## Summary

### Recommended Workflow:

**Before Going to Lab:**
1. ✅ Read WORKSTATION_STEP_BY_STEP.md
2. ✅ Understand the process

**At the Lab:**
1. ✅ Transfer project (10-15 min)
2. ✅ Setup environment (10-15 min)
3. ✅ **Install Claude Code** (5-10 min) ← HIGHLY RECOMMENDED
4. ✅ Open project in Claude Code
5. ✅ Ask Claude Code to verify setup
6. ✅ Ask Claude Code to optimize for workstation GPU
7. ✅ Start training with Claude Code's help
8. ✅ Leave Claude Code open to monitor

**Next Day:**
1. ✅ Use Claude Code to check results
2. ✅ Use Claude Code to start next model
3. ✅ Cleanup with Claude Code's help

---

## Final Recommendation

**YES - Absolutely use Claude Code on the workstation!**

**Benefits:**
- Saves you hours of troubleshooting time
- Automatic project context from CLAUDE.md
- Like having an expert assistant available 24/7
- Optimizes training for the specific workstation GPU
- Prevents common mistakes

**Time Investment:** 5-10 minutes to install and setup

**Time Saved:** Potentially hours if errors occur, plus faster training with optimized configs

**Worth it?** 100% YES! ✅

---

Good luck with your training! Claude Code will be there to help you every step of the way! 🚀
