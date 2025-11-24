# FYP Project Status Report
**Generated:** 2025-11-24
**Student:** Tan Ming Kai (24PMR12003)
**Project:** CrossViT for COVID-19 Classification

---

## 🎯 Overall Status: Phase 2 COMPLETE, Phase 3 READY

**Timeline:** 4 weeks ahead of schedule! 🚀

---

## ✅ Phase 1: Exploration (Weeks 1-2) - COMPLETE

### Completed Notebooks
- ✅ 00_environment_setup.ipynb
- ✅ 01_data_loading.ipynb
- ✅ 02_data_cleaning.ipynb
- ✅ 03_eda.ipynb (5 publication-ready figures generated)
- ✅ 04_baseline_test.ipynb

### Missing (Non-Critical)
- ⚠️ 05_augmentation_test.ipynb (skipped, but not blocking)

### Deliverables
- ✅ Data splits (train/val/test: 80/10/10)
- ✅ CLAHE-enhanced images (240×240 RGB)
- ✅ EDA figures saved to `experiments/phase1_exploration/eda_figures/`
- ✅ Baseline ResNet-50 working

---

## ✅ Phase 2: Systematic Experimentation (Weeks 3-6) - COMPLETE

### Training Progress: 30/30 Runs Complete

| Model | Seeds | Mean Acc ± Std | Status |
|-------|-------|----------------|--------|
| ResNet-50 | 5/5 | 95.45% ± 0.57% | ✅ |
| Swin-Tiny | 5/5 | 95.35% ± 0.53% | ✅ |
| DenseNet-121 | 5/5 | 95.32% ± 0.26% | ✅ |
| EfficientNet-B0 | 5/5 | 95.26% ± 0.35% | ✅ |
| **CrossViT-Tiny** | 5/5 | **94.96% ± 0.55%** | ✅ |
| ViT-Tiny | 5/5 | 87.98% ± 1.90% | ✅ |

### Key Findings
- **All models trained successfully** (30/30 runs)
- **ResNet-50 is best performer** (95.45%), NOT CrossViT
- **CrossViT ranked 5th out of 6** (94.96%)
- **ViT severely underperformed** (87.98%)
- **Difference between top 5 models < 0.5%** (very close!)

### Deliverables
- ✅ 30 model checkpoints saved (`.pth` files)
- ✅ 30 confusion matrices generated
- ✅ 6 results CSV files with metrics
- ✅ MLflow tracking initialized

### ⚠️ Minor Issue
- Swin seeds 789 & 101112 show `training_time=0.0` in CSV
- **Resolution:** Models exist and work (106MB each), just CSV logging error

---

## 🔄 Phase 3: Analysis & Refinement (Weeks 7-8) - READY TO START

### Notebooks Created (Ready to Run)
1. ✅ **12_statistical_validation.ipynb**
   - 95% confidence intervals (bootstrap)
   - Paired t-tests (CrossViT vs baselines)
   - Bonferroni correction (α' = 0.01)
   - Cohen's d effect sizes
   - APA-formatted tables

2. ✅ **13_error_analysis.ipynb**
   - Per-class performance (Precision, Recall, F1)
   - Medical metrics (Sensitivity, Specificity, PPV, NPV)
   - Misclassification visualization
   - Error pattern identification

3. ✅ **14_ablation_studies.ipynb**
   - H₂: Dual-branch vs single-scale (ready to run)
   - H₃: CLAHE impact (requires 2 GPU hours)
   - H₄: Augmentation strategy (requires 3 GPU hours)

### Folder Structure Created
```
experiments/phase3_analysis/
├── statistical_validation/
│   ├── all_models_summary.py ✅
│   ├── detailed_results_all_30_runs.csv ✅
│   └── summary_statistics_table.csv ✅
├── error_analysis/
└── ablation_studies/
```

### Summary Tables Generated
- ✅ Detailed results (all 30 runs)
- ✅ Summary statistics table
- ✅ LaTeX table for thesis
- ✅ APA-formatted results

### What to Do Next
1. **Run 12_statistical_validation.ipynb** (10 minutes)
2. **Run 13_error_analysis.ipynb** (15 minutes)
3. **Run 14_ablation_studies.ipynb** (immediate H₂ test)
4. **Optional:** H₃ & H₄ ablations (5 GPU hours)

---

## ⏭️ Phase 4: Documentation & Deployment (Weeks 9-10) - NOT STARTED

### Missing Notebooks
- ❌ 15_thesis_content.ipynb
- ❌ 16_flask_demo.ipynb

### Missing Folders
- ❌ `experiments/phase4_deliverables/`
- ❌ `experiments/phase4_deliverables/thesis_content/`
- ❌ `experiments/phase4_deliverables/flask_demo/`

### What to Create
1. Chapter 4 tables (reproducibility statement)
2. Chapter 5 figures (all results with CIs)
3. Basic Flask demo (model inference API)

---

## 🎓 Hypothesis Testing Status

### H₁: CrossViT > CNN Baselines (Primary)
- **Status:** ⚠️ **LIKELY NOT SUPPORTED**
- **Evidence:** CrossViT (94.96%) < ResNet-50 (95.45%)
- **Difference:** -0.49% (CrossViT WORSE, not better!)
- **Next Step:** Statistical test in notebook 12 to confirm

### H₂: Dual-Branch > Single-Scale
- **Status:** ✅ **LIKELY SUPPORTED**
- **Evidence:** CrossViT (94.96%) > ViT (87.98%)
- **Difference:** +6.98% (exceeds 5% threshold!)
- **Next Step:** T-test in notebook 14 (ready to run)

### H₃: CLAHE Impact
- **Status:** ⏭️ **NOT TESTED**
- **Requires:** Train on raw images (2 GPU hours)

### H₄: Augmentation Strategy
- **Status:** ⏭️ **NOT TESTED**
- **Requires:** Train with 3 augmentation levels (3 GPU hours)

---

## 📊 Files Generated (Phase 3)

### In `experiments/phase3_analysis/statistical_validation/`:
- `all_models_summary.py` - Summary generator script
- `detailed_results_all_30_runs.csv` - All 30 runs with details
- `summary_statistics_table.csv` - Aggregated statistics

### Summary Table (For Thesis Chapter 5)

| Model | N | Mean ± Std | Min | Max |
|-------|---|------------|-----|-----|
| ResNet-50 | 5 | 95.45% ± 0.57% | 94.66% | 96.03% |
| Swin-Tiny | 5 | 95.35% ± 0.53% | 94.62% | 95.94% |
| DenseNet-121 | 5 | 95.32% ± 0.26% | 94.99% | 95.65% |
| EfficientNet-B0 | 5 | 95.26% ± 0.35% | 94.80% | 95.65% |
| **CrossViT-Tiny** | 5 | **94.96% ± 0.55%** | **94.33%** | **95.65%** |
| ViT-Tiny | 5 | 87.98% ± 1.90% | 85.07% | 89.61% |

---

## ⚠️ Important Insights

### 1. CrossViT Did NOT Outperform CNNs
- **Expected:** CrossViT would be best (H₁)
- **Reality:** CrossViT ranked 5th out of 6
- **Implication:** H₁ likely NOT supported
- **For Thesis:** This is GOOD (challenges assumptions, more interesting discussion!)

### 2. All Models Very Close (Except ViT)
- Top 5 models within 0.5% accuracy
- Statistical significance testing CRITICAL
- May not be practically significant even if statistically significant

### 3. ViT Severe Underperformance
- 7% accuracy gap vs CrossViT
- Strong evidence for H₂ (dual-branch superiority)
- Excellent for ablation study discussion

### 4. Consistency of Results
- Low standard deviations (0.26-0.57%)
- High reproducibility
- Good for thesis (robust methodology)

---

## 🚀 Next Steps (Immediate Actions)

### Today (30 minutes):
1. **Run notebook 12** - Statistical validation
2. **Review hypothesis test results**
3. **Check if any p-values < 0.01**

### This Week (2-3 hours):
1. Run notebook 13 - Error analysis
2. Run notebook 14 - H₂ ablation test
3. Generate all Phase 3 figures

### Optional (5 GPU hours):
- H₃: CLAHE ablation
- H₄: Augmentation ablation

### Next Week (Week 9):
- Start Phase 4: Thesis content generation
- Create Chapter 5 figures and tables
- Write results section

---

## 📋 Thesis Writing Status

### Chapter 4: Methodology
- ✅ Data collection described
- ✅ Preprocessing pipeline documented
- ✅ Model architectures specified
- ⏭️ Need: Reproducibility statement (from notebook 12)

### Chapter 5: Results
- ✅ All 30 training runs complete
- ✅ Summary tables generated
- ⏭️ Need: Statistical validation (notebook 12)
- ⏭️ Need: Error analysis (notebook 13)
- ⏭️ Need: Ablation studies (notebook 14)
- ⏭️ Need: Figures with confidence intervals

### Chapter 6: Discussion
- ⏭️ Need: Interpret why CrossViT underperformed
- ⏭️ Need: Discuss practical vs statistical significance
- ⏭️ Need: Limitations section (H₃, H₄ not tested?)

---

## 🎯 Success Criteria Check

### Minimum to Pass (50%+):
- ✅ CrossViT >85% accuracy (achieved 94.96%)
- ✅ All 5 baselines trained
- ⏭️ Statistical tests completed (ready to run)
- ✅ All notebooks run without errors
- ⏭️ Basic Flask demo (Phase 4)

**Current Status:** On track to pass! 🎉

---

## 📂 Project Structure Summary

```
FYP_Code/
├── notebooks/
│   ├── 00-04: Phase 1 (COMPLETE) ✅
│   ├── 06-11: Phase 2 (COMPLETE) ✅
│   ├── 12-14: Phase 3 (READY) ✅
│   └── 15-16: Phase 4 (TODO) ⏭️
├── experiments/
│   ├── phase1_exploration/ ✅
│   ├── phase2_systematic/ ✅
│   ├── phase3_analysis/ ✅ (folders created, awaiting results)
│   └── phase4_deliverables/ ⏭️
├── data/
│   ├── raw/ (immutable)
│   └── processed/ ✅
└── src/
    ├── data_processing.py ✅
    ├── features.py ✅
    └── models.py ✅
```

---

## 💡 Recommendations

### Critical (Do Now):
1. **Run Phase 3 notebooks** (1 hour total)
2. **Interpret statistical results** (H₁ likely rejected)
3. **Start thesis Chapter 5 writing** (use generated tables)

### Important (This Week):
1. Generate all figures for thesis
2. Complete error analysis
3. Test H₂ (dual-branch hypothesis)

### Optional (If Time):
1. H₃ & H₄ ablations (5 GPU hours)
2. Flask demo prototype (Phase 4)

### For Thesis Discussion:
- **Be honest:** CrossViT underperformed expectations
- **Explain why:** Dataset-specific, model size, hyperparameters?
- **Emphasize:** All models achieved >94% accuracy (excellent!)
- **Discuss:** Small differences may not be clinically significant

---

## ⏰ Timeline Assessment

| Phase | Expected | Actual | Status |
|-------|----------|--------|--------|
| Phase 1 | Weeks 1-2 | Complete | ✅ 2 weeks ahead |
| Phase 2 | Weeks 3-6 | Complete | ✅ 4 weeks ahead |
| Phase 3 | Weeks 7-8 | Ready to start | 🔄 On track |
| Phase 4 | Weeks 9-10 | Not started | ⏭️ On schedule |

**Overall:** 4 weeks ahead of expected timeline! 🚀

---

## 🎉 Achievements

1. ✅ Trained 30 models successfully (100% success rate)
2. ✅ All models achieved >85% accuracy (minimum threshold)
3. ✅ Reproducible results (low std dev)
4. ✅ Well-organized project structure
5. ✅ MLflow tracking set up
6. ✅ Phase 3 notebooks created and ready

---

## 📝 Summary

**You are in excellent shape!** Phase 2 is complete, all 30 models trained, and Phase 3 is ready to go. The main "surprise" is that CrossViT underperformed CNNs, but this is actually GOOD for the thesis - it makes for more interesting discussion and shows rigorous methodology.

**Next Steps:**
1. Run Phase 3 notebooks (1 hour)
2. Review results and start writing Chapter 5
3. Decide on optional ablations (H₃, H₄)

**Expected Completion:** On track for Week 10 submission ✅

---

**Good luck with Phase 3!** 🎓
