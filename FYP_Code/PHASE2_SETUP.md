# Phase 2: Systematic Experimentation - Setup Complete

**Status:** Ready to begin training
**Created:** 2025-11-12
**Student:** Tan Ming Kai (24PMR12003)

## ✅ What Has Been Completed

### 1. Phase 1 Baseline Test (DONE ✅)
- `04_baseline_test.ipynb` - ResNet-50 baseline trained successfully
- **Full dataset results:** 94.76% test accuracy
- Training pipeline verified and working
- GPU memory usage confirmed safe (<7GB)

### 2. Phase 2 Structure Created
**Notebooks created:**
- ✅ `06_crossvit_training.ipynb` - CrossViT-Tiny (PRIMARY MODEL)
- ✅ `07_resnet50_training.ipynb` - ResNet-50 Baseline

**Notebooks remaining to create:**
- ⏳ `08_densenet121_training.ipynb` - DenseNet-121 Baseline
- ⏳ `09_efficientnet_training.ipynb` - EfficientNet-B0 Baseline
- ⏳ `10_vit_training.ipynb` - ViT-Base/16 Baseline
- ⏳ `11_swin_training.ipynb` - Swin-Tiny Baseline

### 3. MLflow Installed
- ✅ MLflow 3.6.0 installed successfully
- Experiment tracking ready: `crossvit-covid19-classification`
- View results: `mlflow ui` → http://localhost:5000

## 📋 Phase 2 Training Plan

### Goal
Train **6 models** × **5 seeds** = **30 total training runs**

### Models & Specifications

| # | Model | Parameters | Batch Size | Learning Rate | Description |
|---|-------|------------|------------|---------------|-------------|
| 1 | **CrossViT-Tiny** | 7M | 8 | 5e-5 | PRIMARY MODEL (dual-branch) |
| 2 | ResNet-50 | 25.6M | 24 | 1e-4 | CNN baseline |
| 3 | DenseNet-121 | 8M | 16 | 1e-4 | Dense connections |
| 4 | EfficientNet-B0 | 5.3M | 16 | 1e-4 | Compound scaling |
| 5 | ViT-Base/16 | 86M | 8 | 5e-5 | Pure transformer |
| 6 | Swin-Tiny | 28M | 12 | 5e-5 | Hierarchical transformer |

### Random Seeds
All models train with: **42, 123, 456, 789, 101112**

## 🚀 How to Proceed

### Option A: Complete Remaining Notebooks First

Since notebooks 08-11 are not yet created, you can:

1. **Copy `07_resnet50_training.ipynb` as a template** and adapt it for each remaining model:

   ```bash
   # For each model, copy and modify:
   cp notebooks/07_resnet50_training.ipynb notebooks/08_densenet121_training.ipynb
   # Then edit the notebook to change:
   # - Title and description
   # - Model loading code
   # - Batch size and learning rate
   # - File names (resnet50 → densenet121)
   ```

2. **Key changes needed per model:**

   **DenseNet-121 (08):**
   ```python
   # Model loading:
   model = models.densenet121(pretrained=True)
   model.classifier = nn.Linear(model.classifier.in_features, config['num_classes'])

   # Config:
   'batch_size': 16,
   'learning_rate': 1e-4,
   ```

   **EfficientNet-B0 (09):**
   ```python
   # Need timm library:
   import timm
   model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=4)

   # Config:
   'batch_size': 16,
   'learning_rate': 1e-4,
   ```

   **ViT-Base/16 (10):**
   ```python
   # Need timm library:
   import timm
   model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=4)

   # Config:
   'batch_size': 8,  # Large model needs smaller batch
   'learning_rate': 5e-5,
   'image_size': 224,  # ViT uses 224x224
   ```

   **Swin-Tiny (11):**
   ```python
   # Need timm library:
   import timm
   model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=4)

   # Config:
   'batch_size': 12,
   'learning_rate': 5e-5,
   'image_size': 224,  # Swin uses 224x224
   ```

### Option B: Start Training with Available Notebooks

You can start Phase 2 training NOW with the notebooks already created:

1. **Start with CrossViT** (most important):
   ```bash
   # Open and run notebooks/06_crossvit_training.ipynb
   # This will train CrossViT with all 5 seeds
   # Estimated time: ~10-15 hours total (2-3 hours per seed)
   ```

2. **Then train ResNet-50** (for comparison):
   ```bash
   # Open and run notebooks/07_resnet50_training.ipynb
   # Estimated time: ~5-8 hours total (1-1.5 hours per seed)
   ```

3. **While training, create remaining notebooks** (08-11)

## 📊 Expected Results

After Phase 2 completion, you will have:

- **30 trained models saved** in `models/` directory
- **30 confusion matrices** in `results/` directory
- **30 MLflow runs** with complete metrics
- **6 results CSV files** (one per model) with statistics

### Example Results Structure:
```
results/
├── crossvit_results.csv       # Mean ± Std across 5 seeds
├── resnet50_results.csv
├── densenet121_results.csv
├── efficientnet_results.csv
├── vit_results.csv
└── swin_results.csv

models/
├── crossvit_best_seed42.pth
├── crossvit_best_seed123.pth
├── ... (30 total model files)
```

## ⏱️ Time Estimates

Based on Phase 1 results (ResNet-50 full dataset: 40 min):

| Model | Time per Seed | Total Time (5 seeds) |
|-------|---------------|----------------------|
| CrossViT-Tiny | 2-3 hours | 10-15 hours |
| ResNet-50 | 1-1.5 hours | 5-8 hours |
| DenseNet-121 | 1-2 hours | 5-10 hours |
| EfficientNet-B0 | 1-2 hours | 5-10 hours |
| ViT-Base/16 | 3-4 hours | 15-20 hours |
| Swin-Tiny | 2-3 hours | 10-15 hours |

**Total Phase 2 time: ~50-80 hours** (can run overnight/over weekends)

## 💡 Tips for Success

1. **Train one seed at a time initially** to verify everything works
2. **Monitor GPU temperature** during long training sessions
3. **Use MLflow UI** to compare runs in real-time
4. **Save intermediate results** after each seed completes
5. **Start with CrossViT and ResNet-50** (most critical for thesis)
6. **Run overnight** for long training jobs

## 🎯 Next Steps

**Immediate (Today):**
1. ✅ You can start training `06_crossvit_training.ipynb` RIGHT NOW
2. While CrossViT trains, create notebooks 08-11 using the template

**This Week:**
1. Complete all 6 model notebooks
2. Train CrossViT and ResNet-50 (most important)
3. Start training remaining baselines

**Next Week (Weeks 3-4):**
1. Complete all 30 training runs
2. Verify all MLflow logs are complete
3. Move to Phase 3: Statistical validation

## 📖 Reference

- **CLAUDE.md** - Complete project specifications
- **Phase 1 notebook:** `04_baseline_test_FULL.ipynb` - Your working example
- **MLflow docs:** https://mlflow.org/docs/latest/index.html

## ❓ Questions to Consider

Before starting Phase 2, decide:

1. **Do you want to create all notebooks first, or start training now?**
   - Recommendation: Start training CrossViT + ResNet-50, create others in parallel

2. **Will you train all seeds sequentially or split across days?**
   - Recommendation: Train 1-2 seeds per day per model

3. **Do you need to test any models on a small subset first?**
   - Recommendation: Run 1 seed with subset to verify, then full dataset

---

**Status:** ✅ Ready to begin Phase 2 systematic experimentation

**Created by:** Claude Code (Assisted by Tan Ming Kai)
