# Phase 2 Training Completion Report

**Generated:** 2025-11-24
**Status:** ✅ 100% COMPLETE (30/30 models trained)

---

## 📊 Executive Summary

All 6 models have been trained with 5 random seeds each, totaling 30 models.
All metrics, checkpoints, and confusion matrices have been verified.

**Phase 2 Status: READY FOR PHASE 3 (STATISTICAL VALIDATION)**

---

## ✅ Verification Checklist

| Item | Expected | Actual | Status |
|------|----------|--------|--------|
| Model Checkpoints (.pth) | 30 | 30 | ✅ Complete |
| Results CSV Files | 6 | 6 | ✅ Complete |
| Seeds per CSV | 5 | 5 | ✅ Complete |
| Confusion Matrices | 30+ | 32 | ✅ Complete |
| ResNet-50 Models | 5 | 5 | ✅ Complete |
| DenseNet-121 Models | 5 | 5 | ✅ Complete |
| EfficientNet-B0 Models | 5 | 5 | ✅ Complete |
| CrossViT-Tiny Models | 5 | 5 | ✅ Complete |
| ViT-Base/16+ Models | 5 | 5 | ✅ Complete |
| Swin-Tiny Models | 5 | 5 | ✅ Complete |

---

## 📈 Complete Results Summary (All Seeds)

### 1. ResNet-50 (Baseline 1)
**Seeds:** 42, 123, 456, 789, 101112

| Seed | Test Accuracy | Test Loss | Training Time (s) |
|------|--------------|-----------|-------------------|
| 42 | 95.28% | 0.1233 | 1,583 |
| 123 | 95.98% | 0.1345 | 2,446 |
| 456 | 94.66% | 0.1428 | 1,789 |
| 789 | 95.28% | 0.1243 | 1,546 |
| 101112 | 96.03% | 0.1390 | 1,896 |

**Mean ± Std:** 95.45% ± 0.51%
**Range:** [94.66%, 96.03%]

---

### 2. DenseNet-121 (Baseline 2)
**Seeds:** 42, 123, 456, 789, 101112

| Seed | Test Accuracy | Test Loss | Training Time (s) |
|------|--------------|-----------|-------------------|
| 42 | 95.28% | 0.1035 | 1,894 |
| 123 | 95.51% | 0.1324 | 2,282 |
| 456 | 95.18% | 0.1106 | 1,833 |
| 789 | 95.65% | 0.1134 | 1,829 |
| 101112 | 94.99% | 0.1149 | 2,054 |

**Mean ± Std:** 95.32% ± 0.27%
**Range:** [94.99%, 95.65%]

---

### 3. EfficientNet-B0 (Baseline 3)
**Seeds:** 42, 123, 456, 789, 101112

| Seed | Test Accuracy | Test Loss | Training Time (s) |
|------|--------------|-----------|-------------------|
| 42 | 95.18% | 0.1344 | 1,790 |
| 123 | 95.56% | 0.1189 | 1,594 |
| 456 | 95.65% | 0.1203 | 1,933 |
| 789 | 94.80% | 0.1411 | 1,996 |
| 101112 | 95.09% | 0.1263 | 1,803 |

**Mean ± Std:** 95.26% ± 0.32%
**Range:** [94.80%, 95.65%]

---

### 4. CrossViT-Tiny (PRIMARY MODEL)
**Seeds:** 42, 123, 456, 789, 101112

| Seed | Test Accuracy | Test Loss | Training Time (s) |
|------|--------------|-----------|-------------------|
| 42 | 94.66% | 0.1121 | 1,983 |
| 123 | 95.42% | 0.1226 | 2,361 |
| 456 | 95.65% | 0.1423 | 2,637 |
| 789 | 94.33% | 0.1230 | 2,461 |
| 101112 | 94.76% | 0.1274 | 2,633 |

**Mean ± Std:** 94.97% ± 0.50%
**Range:** [94.33%, 95.65%]

---

### 5. ViT-Base/16+ (Baseline 4)
**Seeds:** 42, 123, 456, 789, 101112

| Seed | Test Accuracy | Test Loss | Training Time (s) |
|------|--------------|-----------|-------------------|
| 42 | 87.06% | 0.2825 | 4,150 |
| 123 | 89.23% | 0.2667 | 4,898 |
| 456 | 89.61% | 0.2284 | 4,916 |
| 789 | 85.07% | 0.2906 | 4,751 |
| 101112 | 88.95% | 0.2686 | 21,694 |

**Mean ± Std:** 87.98% ± 1.84%
**Range:** [85.07%, 89.61%]

**Note:** ViT shows lower performance, possibly due to requiring larger datasets or longer training.

---

### 6. Swin-Tiny (Baseline 5)
**Seeds:** 42, 123, 456, 789, 101112

| Seed | Test Accuracy | Test Loss | Training Time (s) |
|------|--------------|-----------|-------------------|
| 42 | 94.62% | 0.1136 | 1,900 |
| 123 | 95.09% | 0.1208 | 1,392 |
| 456 | 95.75% | 0.0921 | 1,488 |
| 789 | 95.94% | 0.1170 | 0* |
| 101112 | 95.37% | 0.0999 | 0* |

**Mean ± Std:** 95.35% ± 0.53%
**Range:** [94.62%, 95.94%]

*Training time 0 = recovered from checkpoint (models trained Nov 21)

---

## 🏆 Model Ranking by Mean Accuracy

| Rank | Model | Mean Accuracy | Std Dev | Range |
|------|-------|--------------|---------|-------|
| 1 | ResNet-50 | 95.45% | 0.51% | [94.66%, 96.03%] |
| 2 | Swin-Tiny | 95.35% | 0.53% | [94.62%, 95.94%] |
| 3 | DenseNet-121 | 95.32% | 0.27% | [94.99%, 95.65%] |
| 4 | EfficientNet-B0 | 95.26% | 0.32% | [94.80%, 95.65%] |
| 5 | CrossViT-Tiny | 94.97% | 0.50% | [94.33%, 95.65%] |
| 6 | ViT-Base/16+ | 87.98% | 1.84% | [85.07%, 89.61%] |

---

## 📁 File Locations

### Model Checkpoints (3.5 GB total - NOT on GitHub)
```
experiments/phase2_systematic/models/
├── resnet50/          (5 × ~85 MB = 425 MB)
├── densenet121/       (5 × ~27 MB = 135 MB)
├── efficientnet/      (5 × ~16 MB = 80 MB)
├── crossvit/          (5 × ~28 MB = 140 MB)
├── vit/               (5 × ~327 MB = 1635 MB)
└── swin/              (5 × ~106 MB = 530 MB)
```

### Results Files (ON GitHub ✓)
```
experiments/phase2_systematic/results/
├── metrics/
│   ├── resnet50_results.csv (5 seeds)
│   ├── densenet121_results.csv (5 seeds)
│   ├── efficientnet_results.csv (5 seeds)
│   ├── crossvit_results.csv (5 seeds)
│   ├── vit_results.csv (5 seeds)
│   └── swin_results.csv (5 seeds)
└── confusion_matrices/
    └── 32 PNG files (all seeds + extras)
```

---

## 🎯 Phase 3 Readiness Check

All prerequisites for Phase 3 (Statistical Validation) are met:

- ✅ All 30 models trained with reproducible seeds
- ✅ All results CSV files have complete 5-seed data
- ✅ Confusion matrices generated for all runs
- ✅ Mean accuracies calculated for all models
- ✅ Standard deviations computed
- ✅ All data ready for 95% CI calculations
- ✅ All data ready for paired t-tests
- ✅ All data ready for hypothesis testing (H1, H2, H3, H4)

---

## 📝 Key Observations

1. **CNN Baselines Perform Well:**
   - ResNet-50, DenseNet-121, EfficientNet-B0 all achieve 95.2-95.5% accuracy
   - Very consistent results (std < 0.6%)

2. **CrossViT vs CNNs:**
   - CrossViT (94.97%) performs slightly below CNN baselines
   - May need further investigation in Phase 3 error analysis
   - Hypothesis H1 may need careful statistical testing

3. **ViT Underperforms:**
   - ViT-Base/16+ achieves only 87.98%
   - Much higher variance (1.84%)
   - Likely due to dataset size or training epochs
   - Seed 101112 took 21,694s (6 hours!) vs ~4,500s for others

4. **Swin-Transformer Competitive:**
   - Swin-Tiny performs very well (95.35%)
   - Second-best model overall
   - Shows transformers can work with proper architecture

5. **Seed Stability:**
   - Most models show consistent results across seeds
   - Standard deviations < 0.6% for most models
   - Good reproducibility achieved

---

## 🔬 Next Steps (Phase 3)

1. **Statistical Validation** (`12_statistical_validation.ipynb`)
   - Calculate 95% confidence intervals for all models
   - Perform paired t-tests (CrossViT vs each baseline)
   - Apply Bonferroni correction (α' = 0.05/5 = 0.01)
   - Test hypothesis H1: CrossViT > baselines (p < 0.05)

2. **Error Analysis** (`13_error_analysis.ipynb`)
   - Analyze misclassifications per model
   - Per-class performance breakdown
   - Identify failure patterns
   - Compare error types across architectures

3. **Ablation Studies** (`14_ablation_studies.ipynb`)
   - Test H2: Dual-branch vs single-scale
   - Test H3: CLAHE vs no CLAHE
   - Test H4: Conservative augmentation impact

---

## ⚠️ Important Notes

1. **Model Files Not on GitHub:**
   - 30 .pth files (3.5 GB) are in .gitignore
   - Stored locally at: `C:\Users\FOCS3\Documents\GitHub\fyp-project\FYP_Code\experiments\phase2_systematic\models\`
   - **Must backup to USB before major system changes**

2. **Swin CSV Recovery:**
   - Seeds 789 and 101112 were recovered using `evaluate_missing_swin_seeds.py`
   - Original training data from Nov 21 was lost when CSV was overwritten Nov 24
   - Recovery script evaluates saved checkpoints to regenerate metrics

3. **ViT Seed 101112 Anomaly:**
   - Took 21,694s (6 hours) vs 4,500s for other seeds
   - Reason unknown - may have hit memory issues or other bottleneck
   - Results (88.95%) are consistent with other ViT seeds

4. **Resolution Differences:**
   - All models use 240×240 except Swin (256×256)
   - Swin-Tiny requires 256×256 (no 240 variant available in timm)
   - Fair comparison maintained as difference is minimal

---

## 📊 Statistics for Thesis Chapter 5

**Total Training:**
- 30 models trained
- 5 random seeds: [42, 123, 456, 789, 101112]
- Total training time: ~22 hours
- Total model size: 3.5 GB

**Performance Summary:**
- Best single result: ResNet-50 seed 101112 (96.03%)
- Worst single result: ViT seed 789 (85.07%)
- Best mean performance: ResNet-50 (95.45%)
- Most consistent: DenseNet-121 (std = 0.27%)
- Least consistent: ViT-Base/16+ (std = 1.84%)

**Hypothesis H1 Preliminary Assessment:**
- CrossViT (94.97%) < ResNet-50 (95.45%) by 0.48%
- CrossViT (94.97%) < Swin-Tiny (95.35%) by 0.38%
- CrossViT (94.97%) < DenseNet-121 (95.32%) by 0.35%
- Statistical significance testing required (Phase 3)

---

**Report Generated:** 2025-11-24
**Phase 2 Status:** ✅ COMPLETE
**Next Phase:** Phase 3 - Statistical Validation

---

END OF REPORT
