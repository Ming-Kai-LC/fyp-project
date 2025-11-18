# Phase-Based Folder Structure - Enforcement Guide

**Purpose:** Ensure Claude ALWAYS follows phase-based folder organization in future sessions

---

## Enforcement Mechanisms

### 1. CLAUDE.md (Project Instructions) ✅
- **Location:** `CLAUDE.md` (root directory)
- **Status:** ✅ Updated with phase-based structure
- **What it does:** Every Claude session reads CLAUDE.md at startup
- **Key sections:**
  - "Code Organization (Phase-Based Structure)" - lines 285-328
  - Lists all phase folders and their purposes
  - Explicitly instructs to use `@fyp-folder-structure` skill

### 2. @fyp-folder-structure Skill ✅
- **Location:** `.claude/skills/fyp-folder-structure/SKILL.md`
- **Status:** ✅ Created and committed to git
- **What it does:** Provides detailed rules for file placement
- **Invoke with:** `@fyp-folder-structure` in any message
- **Contains:**
  - Complete folder structure
  - 10 file location rules
  - Code examples for each phase
  - Quick reference table
  - Common mistakes to avoid

### 3. Updated Code ✅
- **File:** `train_all_models_safe.py`
- **Status:** ✅ Already using new paths
- **What it does:** Training scripts automatically save to correct locations
- **Paths configured:**
  ```python
  MODELS_DIR = "experiments/phase2_systematic/models"
  RESULTS_DIR = "experiments/phase2_systematic/results/confusion_matrices"
  mlflow.set_tracking_uri("file:./experiments/phase2_systematic/mlruns")
  ```

### 4. experiments/README.md ✅
- **Location:** `experiments/README.md`
- **Status:** ✅ Created with structure documentation
- **What it does:** Quick reference for anyone working in experiments/
- **Contains:**
  - Phase breakdown (1-4)
  - Expected outputs per phase
  - Usage instructions

---

## How Future Claude Will Enforce This

### Automatic Enforcement (No User Action Needed)

1. **On Session Start:**
   - Claude reads `CLAUDE.md` → sees phase-based structure instructions
   - Sees: "IMPORTANT: Always use `@fyp-folder-structure` skill"
   - Sees: "ALWAYS DO: Save all outputs to `experiments/phase{1-4}/` folders"

2. **When Creating Files:**
   - Claude checks CLAUDE.md for correct location
   - Consults `@fyp-folder-structure` skill if unsure
   - Uses phase-specific paths automatically

3. **When Running Scripts:**
   - `train_all_models_safe.py` already configured with new paths
   - MLflow automatically uses `experiments/phase2_systematic/mlruns/`
   - No manual path specification needed

### Manual Enforcement (User Can Invoke)

**If user suspects wrong location:**
```
User: "Where should I save this file?"
Claude: *Checks @fyp-folder-structure skill*
Claude: "Based on Phase 2, save to: experiments/phase2_systematic/results/..."
```

**If user wants verification:**
```
User: "@fyp-folder-structure check this path: models/resnet50.pth"
Claude: "❌ WRONG: Old structure. Should be: experiments/phase2_systematic/models/resnet50/resnet50.pth"
```

---

## Verification Checklist

Before ANY file operation, Claude should verify:

- [ ] **Read CLAUDE.md**: Check "Code Organization (Phase-Based Structure)" section
- [ ] **Identify Phase**: Which phase does this output belong to? (1, 2, 3, or 4)
- [ ] **Consult Skill**: If unsure, invoke `@fyp-folder-structure`
- [ ] **Check Path**: Does path start with `experiments/phase{X}/`?
- [ ] **Verify Type**: Is file in correct subdirectory? (models/, results/, etc.)

---

## Git Commit Strategy

### When to Commit

**User explicitly requests:**
```
User: "commit the changes"
User: "push to github"
User: "create a commit"
```

**After major milestones:**
- Completing a phase (e.g., all Phase 1 notebooks done)
- Training all models (30 runs complete)
- Folder reorganization (like this one)
- Adding new features/scripts

### What NOT to Commit Automatically

- ❌ During exploratory work
- ❌ Intermediate training checkpoints (unless requested)
- ❌ Temporary test files
- ❌ Work-in-progress code

### Commit Message Format

Always use detailed commit messages following this template:

```bash
git commit -m "$(cat <<'EOF'
[Short summary in imperative mood]

[Detailed explanation of what changed and why]

KEY CHANGES:
- [Bullet point 1]
- [Bullet point 2]

BENEFITS:
- [Benefit 1]
- [Benefit 2]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Quick Reference: Where Files Go

| File Type | Phase | Path |
|-----------|-------|------|
| **Notebooks** | All | `notebooks/00-16.ipynb` (sequential) |
| **EDA figures** | 1 | `experiments/phase1_exploration/eda_figures/` |
| **Baseline results** | 1 | `experiments/phase1_exploration/baseline_results/` |
| **Augmentation tests** | 1 | `experiments/phase1_exploration/augmentation_tests/` |
| **Model checkpoints** | 2 | `experiments/phase2_systematic/models/{model_name}/` |
| **Confusion matrices** | 2 | `experiments/phase2_systematic/results/confusion_matrices/` |
| **Metrics CSVs** | 2 | `experiments/phase2_systematic/results/metrics/` |
| **Training logs** | 2 | `experiments/phase2_systematic/results/training_logs/` |
| **MLflow** | 2 | `experiments/phase2_systematic/mlruns/` |
| **Statistical validation** | 3 | `experiments/phase3_analysis/statistical_validation/` |
| **Error analysis** | 3 | `experiments/phase3_analysis/error_analysis/` |
| **Ablation studies** | 3 | `experiments/phase3_analysis/ablation_studies/` |
| **Thesis tables** | 4 | `experiments/phase4_deliverables/thesis_content/chapter4_tables/` |
| **Thesis figures** | 4 | `experiments/phase4_deliverables/thesis_content/chapter5_figures/` |
| **Flask demo** | 4 | `experiments/phase4_deliverables/flask_demo/` |
| **Utility scripts** | - | `scripts/` |
| **Documentation** | - | `docs/` |
| **Logs** | - | `logs/` |

---

## Testing the Enforcement

**Test 1: Ask Claude to save a confusion matrix**
```
User: "Save the ResNet-50 confusion matrix for seed 42"
Expected: experiments/phase2_systematic/results/confusion_matrices/resnet50_cm_seed42.png
```

**Test 2: Ask Claude to check a path**
```
User: "Is this path correct? models/resnet50_best.pth"
Expected: ❌ No, should be: experiments/phase2_systematic/models/resnet50/resnet50_best_seed42.pth
```

**Test 3: Ask Claude where to save something**
```
User: "Where do I save EDA class distribution figure?"
Expected: experiments/phase1_exploration/eda_figures/class_distribution.png
```

---

## Summary

### ✅ What's Enforced Automatically

1. **CLAUDE.md** tells every new Claude session to use phase-based structure
2. **@fyp-folder-structure skill** provides detailed rules (committed to git)
3. **Code already updated** to use new paths (train_all_models_safe.py)
4. **experiments/README.md** documents structure for quick reference

### ✅ What User Can Do

1. **Invoke skill:** `@fyp-folder-structure` to check paths
2. **Ask questions:** "Where should I save X?" → Claude checks skill
3. **Request commits:** "commit changes" → Claude commits with detailed message

### ✅ What's Synced to GitHub

- ✅ `.claude/skills/fyp-folder-structure/` (structure rules)
- ✅ `CLAUDE.md` (updated with phase-based organization)
- ✅ `experiments/` (entire new folder structure)
- ✅ `train_all_models_safe.py` (updated paths)

**Result:** Any workstation that clones this repo will have:
- Same folder structure
- Same enforcement rules
- Same Claude instructions
- Same file organization system

---

## Failure Recovery

**If Claude uses wrong path:**
1. User points it out
2. Claude checks `@fyp-folder-structure` skill
3. Claude corrects the path
4. Claude moves file to correct location

**If structure is broken:**
1. Read this file (.claude/STRUCTURE_ENFORCEMENT.md)
2. Verify CLAUDE.md is intact
3. Check `@fyp-folder-structure` skill exists
4. Reconstruct from experiments/README.md

---

**Last Updated:** 2025-11-18
**Commit:** 2b1f7cd - "Implement phase-based folder structure for FYP organization"
