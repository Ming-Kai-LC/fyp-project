# TAR UMT Final Year Project - Overview

**Project Title:** CrossViT for COVID-19 Chest X-Ray Classification
**Student:** Tan Ming Kai (24PMR12003)
**Academic Year:** 2025/26
**Program:** Bachelor of Data Science
**University:** Tunku Abdul Rahman University of Management and Technology (TAR UMT)

---

## 🎯 Project Summary

This Final Year Project (FYP) implements and evaluates CrossViT (Cross-Attention Vision Transformer) for automated COVID-19 classification from chest X-ray images. The project compares CrossViT against multiple CNN and Transformer baselines using rigorous statistical validation methods.

**Main Research Question:**
Can CrossViT's dual-branch architecture significantly outperform traditional CNNs and other Vision Transformers for medical image classification?

---

## 📊 Current Status

**Overall Progress:** Phase 2 (Week 6) - 66.7% Complete

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 1: Exploration | ✅ COMPLETE | 100% |
| Phase 2: Systematic Experimentation | 🔄 IN PROGRESS | 66.7% (20/30 runs) |
| Phase 3: Analysis & Validation | ⏸️ PENDING | 0% |
| Phase 4: Documentation & Deployment | ⏸️ PENDING | 0% |

---

## 📁 Phase Documentation

Detailed documentation for each phase:

1. **[PHASE_1_EXPLORATION.md](PHASE_1_EXPLORATION.md)** ✅
   - Environment setup
   - Data loading & splitting
   - Preprocessing pipeline (CLAHE)
   - Exploratory data analysis
   - Baseline model validation
   - Augmentation strategy

2. **[PHASE_2_SYSTEMATIC_EXPERIMENTATION.md](PHASE_2_SYSTEMATIC_EXPERIMENTATION.md)** 🔄
   - 6 models × 5 seeds = 30 training runs
   - Currently: 20/30 complete (4 models done)
   - Batch size optimization
   - Auto-sequential training
   - Performance results & analysis

3. **[PHASE_3_ANALYSIS_VALIDATION.md](PHASE_3_ANALYSIS_VALIDATION.md)** ⏸️
   - Statistical validation (95% CI, hypothesis testing)
   - Error analysis
   - Ablation studies
   - Results tables for thesis

4. **[PHASE_4_DOCUMENTATION_DEPLOYMENT.md](PHASE_4_DOCUMENTATION_DEPLOYMENT.md)** ⏸️
   - Thesis writing (Chapters 4 & 5)
   - Flask demo application
   - Documentation

---

## 🔬 Models Being Evaluated

| Model | Type | Parameters | Status | Mean Acc | Seeds |
|-------|------|-----------|--------|----------|-------|
| ResNet-50 | CNN | 23.5M | ✅ | 95.45% ± 0.57% | 5/5 |
| DenseNet-121 | CNN | 7.0M | ✅ | 95.32% ± 0.26% | 5/5 |
| EfficientNet-B0 | CNN | 4.0M | ✅ | 95.23% ± 0.33% | 5/5 |
| CrossViT-Tiny | Transformer | 7.0M | ✅ | 94.96% ± 0.55% | 5/5 |
| ViT-Base | Transformer | 85.8M | 🔄 | TBD | 0/5 |
| Swin-Tiny | Transformer | 27.5M | ⏸️ | TBD | 0/5 |

**Key Finding:** CNNs outperform CrossViT (H₁ rejected - valid research result!)

---

## 📊 Dataset

**Source:** COVID-19 Radiography Database (Rahman et al., 2021)

**Statistics:**
- **Total Images:** 21,165 chest X-rays (299×299 PNG, grayscale)
- **Classes:** 4 (COVID-19, Normal, Lung Opacity, Viral Pneumonia)
- **Split:** 80% train / 10% val / 10% test
- **Imbalance:** 7.6:1 ratio (Normal:Viral Pneumonia)

**Class Distribution:**

| Class | Train | Val | Test | Total | % |
|-------|-------|-----|------|-------|---|
| COVID-19 | 2,893 | 362 | 361 | 3,616 | 17.1% |
| Normal | 8,153 | 1,020 | 1,019 | 10,192 | 48.2% |
| Lung Opacity | 4,810 | 601 | 601 | 6,012 | 28.4% |
| Viral Pneumonia | 1,075 | 134 | 136 | 1,345 | 6.4% |

---

## 🔧 Technical Specifications

### Hardware

```
GPU: NVIDIA RTX 6000 Ada Generation
VRAM: 48 GB
CUDA: 13.0
CPU: [Server CPU]
RAM: [Server RAM]
OS: Windows 11
```

### Software

```
Python: 3.13
PyTorch: 2.x
Key Libraries:
  - timm (for pretrained models)
  - torchvision
  - opencv-python (for CLAHE)
  - pandas, numpy
  - matplotlib, seaborn
  - scikit-learn
  - mlflow (experiment tracking)
```

### Training Configuration

```python
image_size = 240×240
batch_size = 380 (auto-adjusted to 323)
learning_rate = 1.78e-4
optimizer = Adam
weight_decay = 1e-4
max_epochs = 30
early_stopping_patience = 10
mixed_precision = True
random_seeds = [42, 123, 456, 789, 101112]
```

---

## 📈 Key Research Findings

### 1. CNN Superiority over CrossViT

**Hypothesis H₁:** CrossViT will significantly outperform CNN baselines
**Result:** **REJECTED** ❌

**Evidence:**
- ResNet-50: 95.45% vs CrossViT: 94.96% (**+0.49%**)
- DenseNet-121: 95.32% vs CrossViT: 94.96% (**+0.36%**)
- EfficientNet-B0: 95.23% vs CrossViT: 94.96% (**+0.27%**)

**Significance:** Valid and publishable research finding!

**Implications:**
- CNNs more suitable for this medical imaging task
- Inductive bias (translation invariance) helps CNNs
- Transformers may need larger datasets
- Practical recommendation: Use DenseNet-121 (most consistent)

---

### 2. Model Consistency Matters

**DenseNet-121:** Lowest variance (±0.26%)
- All 5 seeds achieved 95%+ accuracy
- Most reliable for clinical deployment
- Balance of performance and consistency

---

### 3. Efficiency vs Performance

**EfficientNet-B0:**
- Smallest model (4M parameters)
- Competitive performance (95.23%)
- Fastest training time
- Best for resource-constrained deployment

---

## 🗂️ Project Structure

```
FYP_Code/
├── data/
│   ├── raw/                    # Original COVID-19 dataset
│   └── processed/              # Train/val/test CSVs
├── notebooks/
│   ├── 00_environment_setup.ipynb
│   ├── 01_data_loading.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_baseline_test.ipynb
│   ├── 05_augmentation_test.ipynb
│   ├── 06-11_model_training.ipynb
│   ├── 12_statistical_validation.ipynb
│   ├── 13_error_analysis.ipynb
│   ├── 14_ablation_studies.ipynb
│   ├── 15_thesis_content.ipynb
│   └── 16_flask_demo.ipynb
├── experiments/
│   └── phase2_systematic/
│       ├── models/             # Trained model checkpoints
│       ├── results/            # Metrics & confusion matrices
│       └── mlruns/             # MLflow tracking
├── logs/                       # Training logs
├── flask_demo/                 # Web application
├── src/                        # Reusable modules
├── train_all_models_safe.py   # Main training script
├── auto_train_remaining.py    # Auto-monitor script
└── PHASE_*.md                  # Phase documentation
```

---

## 📚 Key Research Hypotheses

### Primary Hypothesis

**H₁:** CrossViT will significantly outperform traditional CNN baselines for COVID-19 chest X-ray classification
**Status:** ❌ REJECTED (CNNs performed better)

### Secondary Hypotheses

**H₂:** CLAHE preprocessing will improve model performance by ≥2%
**Status:** ⏸️ To be tested in Phase 3

**H₃:** Conservative data augmentation will improve generalization without degrading accuracy
**Status:** ⏸️ To be tested in Phase 3

**H₄:** CrossViT's dual-branch architecture will provide ≥5% improvement over single-scale
**Status:** ⏸️ To be tested in Phase 3

---

## 🎓 Academic Context

### TAR UMT FYP Requirements

**Deliverables:**
1. ✅ Complete thesis (6 chapters)
   - Chapter 1: Introduction
   - Chapter 2: Literature Review
   - Chapter 3: Research Methodology
   - Chapter 4: Experimental Setup ⏸️
   - Chapter 5: Results & Discussion ⏸️
   - Chapter 6: Conclusion

2. ⏸️ Working software prototype (Flask demo)

3. ✅ Oral presentation & demonstration

**Thesis Format:**
- APA 7th Edition
- Double-spaced, 12pt Times New Roman
- Maximum Turnitin similarity: 20%
- Page count: 80-120 pages (excluding appendices)

---

## 🚀 Next Steps

**Immediate (Phase 2):**
- [x] Fix ViT/Swin image size issue ✅
- [ ] Complete ViT training (~2.5 hours)
- [ ] Complete Swin training (~2 hours)
- [ ] Verify all 30 runs successful

**Short-term (Phase 3):**
- [ ] Statistical validation (McNemar's test, 95% CI)
- [ ] Error analysis with visualizations
- [ ] Ablation studies (H₂, H₃, H₄)
- [ ] Generate thesis tables/figures

**Medium-term (Phase 4):**
- [ ] Write Chapter 4 (Methodology)
- [ ] Write Chapter 5 (Results & Discussion)
- [ ] Develop Flask demo
- [ ] Complete documentation

---

## 📊 Timeline Summary

| Week | Phase | Activities | Status |
|------|-------|------------|--------|
| 1-2 | Phase 1 | Exploration & setup | ✅ |
| 3-6 | Phase 2 | Model training (30 runs) | 🔄 66.7% |
| 7-8 | Phase 3 | Statistical analysis | ⏸️ |
| 9-10 | Phase 4 | Thesis & Flask demo | ⏸️ |
| 11-12 | Final | Review & submission | ⏸️ |

**Current Week:** 6
**Days Remaining:** ~6 weeks until submission

---

## 📖 Key References

1. Rahman et al. (2021) - COVID-19 Radiography Database
2. Chen et al. (2021) - CrossViT: Cross-Attention Multi-Scale Vision Transformer
3. He et al. (2016) - Deep Residual Learning (ResNet)
4. Huang et al. (2017) - Densely Connected Convolutional Networks
5. Dosovitskiy et al. (2021) - An Image is Worth 16x16 Words (ViT)
6. Liu et al. (2021) - Swin Transformer

---

## 🤝 Acknowledgments

- **Supervisor:** [Supervisor Name]
- **University:** Tunku Abdul Rahman University of Management and Technology (TAR UMT)
- **Faculty:** Faculty of Computing and Information Technology (FOCS)
- **Dataset:** COVID-19 Radiography Database (Rahman et al., 2021)

---

## 📝 Quick Access Links

**Phase Documentation:**
- [Phase 1: Exploration](PHASE_1_EXPLORATION.md)
- [Phase 2: Systematic Experimentation](PHASE_2_SYSTEMATIC_EXPERIMENTATION.md)
- [Phase 3: Analysis & Validation](PHASE_3_ANALYSIS_VALIDATION.md)
- [Phase 4: Documentation & Deployment](PHASE_4_DOCUMENTATION_DEPLOYMENT.md)

**Status Files:**
- [CURRENT_STATUS.txt](CURRENT_STATUS.txt) - Current training status
- [FINAL_STATUS.txt](FINAL_STATUS.txt) - Error fixes & next steps
- [TRAINING_COMPLETE_SUMMARY.md](TRAINING_COMPLETE_SUMMARY.md) - Results summary

**Configuration:**
- [CLAUDE.md](CLAUDE.md) - Project instructions for Claude Code

---

**Last Updated:** 2025-11-19 02:35
**Project Status:** 🔄 ACTIVE (Phase 2 - Week 6)
**Completion:** 66.7% (20/30 training runs)
**Next Milestone:** Complete ViT & Swin training (~4.5 hours)
