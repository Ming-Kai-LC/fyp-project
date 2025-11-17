# CrossViT COVID-19 Classification - FYP Project

**Student:** Tan Ming Kai (24PMR12003)
**Institution:** Tunku Abdul Rahman University of Management and Technology (TAR UMT)
**Academic Year:** 2025/26
**Project:** Cross-Attention Vision Transformer (CrossViT) for COVID-19 Chest X-ray Classification

---

## Quick Start for Remote Desktop / New Machine Setup

### Prerequisites

- Python 3.8+ with CUDA support
- NVIDIA GPU with 8GB+ VRAM (RTX 4060 or better)
- Windows 10/11 or Linux
- 3GB free disk space (without raw dataset)

### 1. Clone or Download This Repository

**Option A: Clone with Git**
```bash
git clone https://github.com/Ming-Kai-LC/fyp-project.git
cd fyp-project/FYP_Code
```

**Option B: Download ZIP**
1. Click green "Code" button on GitHub
2. Select "Download ZIP"
3. Extract to your desired location
4. Navigate to the `FYP_Code` folder

### 2. Download the Dataset (One-Time Setup)

**⚠️ IMPORTANT:** The raw dataset is NOT included in this repository (877 MB, 21,165 images)

**Option 1: You Already Have Processed Data** ✅ RECOMMENDED
- If you have `data/processed/clahe_enhanced/` folder with 21,165 images
- Skip dataset download, you're ready to train!

**Option 2: Download Raw Dataset from Kaggle**
```bash
# Visit: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
# Download and extract to: data/raw/COVID-19_Radiography_Dataset/
# Then run notebooks 01-02 to create processed data
```

**Option 3: Transfer from Another Machine**
- Copy `data/processed/` folder from your existing setup
- This includes CLAHE-enhanced images and CSV splits

### 3. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Verify GPU is available
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

**Expected Output:**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 4060 [or your GPU name]
```

### 4. Verify Data Integrity

```bash
# Check processed images exist
python -c "import os; count = sum([len([f for f in files if f.endswith('.png')]) for root, dirs, files in os.walk('data/processed/clahe_enhanced')]); print(f'Total images: {count}')"
# Expected: Total images: 21165

# Check CSV splits exist
python -c "import os; csvs = [f for f in os.listdir('data/processed') if f.endswith('.csv')]; print(f'CSV files: {len(csvs)}'); print('\n'.join(csvs))"
# Expected: 7 CSV files (train.csv, val.csv, test.csv, etc.)
```

### 5. Start Training

**Option A: Using Jupyter Notebooks (Recommended)**
```bash
cd notebooks
jupyter notebook

# Open and run notebooks in order:
# - 06_crossvit_training.ipynb (Primary model)
# - 08_densenet121_training.ipynb
# - 09_efficientnet_training.ipynb
# - 10_vit_training.ipynb
# - 11_swin_training.ipynb
```

**Option B: Using Python Scripts**
```bash
cd notebooks
jupyter nbconvert --to script 06_crossvit_training.ipynb
python 06_crossvit_training.py > crossvit_training.log 2>&1
```

---

## Project Structure

```
FYP_Code/
├── data/
│   ├── raw/                    # Original dataset (NOT in Git)
│   ├── processed/              # CLAHE-enhanced images + CSV splits
│   │   ├── clahe_enhanced/     # 21,165 preprocessed images
│   │   ├── train.csv           # Training split (80%)
│   │   ├── val.csv             # Validation split (10%)
│   │   └── test.csv            # Test split (10%)
│   └── external/               # Third-party data
│
├── notebooks/                  # Jupyter notebooks (main workflow)
│   ├── 00_environment_setup.ipynb
│   ├── 01_data_loading.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_baseline_test.ipynb
│   ├── 06_crossvit_training.ipynb    # PRIMARY MODEL
│   ├── 07_resnet50_training.ipynb
│   ├── 08_densenet121_training.ipynb
│   ├── 09_efficientnet_training.ipynb
│   ├── 10_vit_training.ipynb
│   └── 11_swin_training.ipynb
│
├── src/                        # Reusable Python modules
│   ├── data_processing.py
│   ├── features.py
│   └── models.py
│
├── models/                     # Trained model checkpoints (.pth files)
├── results/                    # Confusion matrices, plots, CSVs
├── .claude/                    # Claude Code skills (project context)
│
├── CLAUDE.md                   # 🔥 PROJECT INSTRUCTIONS - READ THIS!
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── *.md                        # Documentation files
```

---

## What's Included vs. What's NOT

### ✅ Included in GitHub

- All Jupyter notebooks (00-11)
- Source code (`src/` modules)
- Processed data CSV splits (7 files, <1 MB)
- Documentation (CLAUDE.md, guides, etc.)
- Claude Code skills (`.claude/` directory)
- Requirements.txt
- .gitignore configuration

### ❌ NOT Included (Too Large for Git)

- Raw dataset (`data/raw/`, 877 MB)
- Processed images (`data/processed/clahe_enhanced/`, 2 GB)
- Trained models (`models/*.pth`, 400+ MB)
- MLflow experiment logs (`notebooks/mlruns/`)
- Virtual environment (`venv/`)
- Training logs (`*.log`)

**How to Get Missing Data:**
1. **Processed Images:** Transfer from your existing setup OR download raw dataset and run notebooks 01-02
2. **Trained Models:** Train models yourself OR download from project releases (if available)
3. **MLflow Logs:** Will be regenerated when you train

---

## Hardware Requirements

### Minimum (Your Laptop Setup)
- GPU: NVIDIA RTX 4060 (8GB VRAM)
- Batch Size: 8 (with gradient accumulation)
- Training Time: ~8 hours per model (5 seeds)

### Recommended (Lab Workstation)
- GPU: RTX 4090 or better (24GB+ VRAM)
- Batch Size: 24-32 (3x faster)
- Training Time: ~3 hours per model (5 seeds)

**Memory Optimization:**
- Mixed precision training enabled (reduces VRAM by 40%)
- Gradient accumulation (effective batch size = 32)
- Persistent workers for DataLoader
- Automatic CUDA cache clearing every 10 batches

---

## Training Configuration

All models use these hyperparameters (as specified in CLAUDE.md):

```python
CONFIG = {
    'batch_size': 8,                    # Adjust based on GPU VRAM
    'gradient_accumulation_steps': 4,   # Effective batch = 32
    'learning_rate': 5e-5,
    'weight_decay': 0.05,
    'max_epochs': 50,
    'early_stopping_patience': 15,
    'random_seeds': [42, 123, 456, 789, 101112],  # 5 seeds for reproducibility
    'num_workers': 4,
    'mixed_precision': True,
}
```

**For Faster GPUs (24GB+ VRAM):**
```python
CONFIG = {
    'batch_size': 24,                   # 3x larger
    'gradient_accumulation_steps': 1,   # No accumulation needed
    # ... rest same
}
```

---

## Models & Expected Results

| Model | Parameters | Input Size | Expected Accuracy | Training Time (8GB) |
|-------|-----------|------------|-------------------|---------------------|
| **CrossViT-Tiny** | 7M | 240×240 | ~94% | 8 hours |
| ResNet-50 | 25M | 224×224 | 95.49% ± 0.33% | 4 hours |
| DenseNet-121 | 8M | 224×224 | ~94% | 4 hours |
| EfficientNet-B0 | 5M | 224×224 | ~93% | 4 hours |
| ViT-Base/16 | 86M | 224×224 | ~92% | 7 hours |
| Swin-Tiny | 28M | 224×224 | ~93% | 6 hours |

**Total Training Time:**
- 8GB GPU: ~30 hours (all 5 models × 5 seeds each)
- 24GB GPU: ~12 hours (with optimized batch sizes)

---

## Using Claude Code (Highly Recommended!)

This project includes `.claude/` skills that provide context-aware assistance.

### Why Use Claude Code?

1. **Automatic Project Understanding**
   - Reads `CLAUDE.md` automatically
   - Knows all model specs, hardware constraints, training sequences
   - Understands your FYP requirements

2. **Real-Time Help**
   - Fix CUDA out-of-memory errors instantly
   - Optimize batch sizes for your GPU
   - Monitor training progress
   - Debug issues without googling

3. **Smart Workflow Guidance**
   - Tells you which notebook to run next
   - Validates results and generates thesis tables
   - Helps with statistical validation

### How to Use

1. **Install Claude Code:**
   - Visit: https://claude.com/code
   - Download and install for your OS
   - Login with your Anthropic account

2. **Open This Project:**
   - Open Claude Code
   - File → Open Folder
   - Select this `FYP_Code` directory

3. **Claude Code Automatically Reads CLAUDE.md!**
   - It will understand your entire project context
   - Just ask questions like:
     - "Check if GPU is available"
     - "What should I train next?"
     - "I got this error: [paste error]"
     - "Optimize training for RTX 4090"

See `USING_CLAUDE_CODE_ON_WORKSTATION.md` for detailed examples.

---

## Remote Desktop Setup (Lab Workstation)

**If you're using this project on a remote desktop workstation:**

1. **Read These Guides:**
   - `WORKSTATION_STEP_BY_STEP.md` - Complete checklist
   - `REMOTE_DESKTOP_TRANSFER_READY.md` - What to expect
   - `USING_CLAUDE_CODE_ON_WORKSTATION.md` - Claude Code help

2. **Quick Transfer Process:**
   ```
   1. Connect to lab Wi-Fi (AINexus24)
   2. Remote Desktop to workstation (metacode1/metacode2)
   3. Copy FYP_Code folder via \\tsclient\D\...
   4. Setup environment (15 minutes)
   5. Start training (runs overnight)
   6. Collect results next day
   7. CLEANUP workstation (mandatory!)
   ```

3. **Time Estimates:**
   - File transfer: 10-15 minutes (2.4 GB)
   - Environment setup: 10-15 minutes
   - Training: 6-8 hours (overnight)

---

## MLflow Experiment Tracking

All training runs are logged with MLflow:

```bash
# View experiment results
cd notebooks
mlflow ui

# Open browser: http://localhost:5000
```

**What MLflow Tracks:**
- All hyperparameters (learning rate, batch size, seeds)
- Training metrics (accuracy, loss per epoch)
- Final test results (accuracy, precision, recall)
- Confusion matrices and plots
- Model checkpoints

**Export Results for Thesis:**
```python
import mlflow
import pandas as pd

experiment = mlflow.get_experiment_by_name("crossvit-covid19-classification")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Export to CSV
runs.to_csv('all_results.csv')
```

---

## Troubleshooting

### Problem: GPU Not Detected

```bash
# Check CUDA installation
python -c "import torch; print(torch.version.cuda)"

# Check GPU driver
nvidia-smi
```

**Solution:** Install/update NVIDIA drivers and CUDA toolkit

### Problem: Out of Memory (CUDA OOM)

**Solution 1:** Reduce batch size
```python
CONFIG['batch_size'] = 4  # Instead of 8
```

**Solution 2:** Enable gradient accumulation
```python
CONFIG['gradient_accumulation_steps'] = 8  # Effective batch = 32
```

### Problem: Data Not Found

**Solution:** Check paths in CSV files
```python
import pandas as pd
df = pd.read_csv('data/processed/train.csv')
print(df['filepath'].head())

# Update paths if needed (e.g., after transferring machines)
df['filepath'] = df['filepath'].str.replace('D:/old/path', 'C:/new/path')
df.to_csv('data/processed/train.csv', index=False)
```

### Problem: Slow Training

**Solution 1:** Increase batch size (if GPU allows)
**Solution 2:** Use more DataLoader workers
```python
CONFIG['num_workers'] = 8  # Instead of 4
```

---

## Next Steps After Download

### First-Time Setup (Choose One Path)

**Path A: You Have Processed Data Already**
```bash
1. Clone/download repository ✓
2. Copy data/processed/ from old setup
3. Setup environment (pip install -r requirements.txt)
4. Verify GPU and data
5. Start training!
```

**Path B: Starting from Raw Dataset**
```bash
1. Clone/download repository ✓
2. Download raw dataset from Kaggle
3. Setup environment (pip install -r requirements.txt)
4. Run notebooks 01-02 to process data (~30 min)
5. Run notebook 04 to test baseline (~1 hour)
6. Start Phase 2 training!
```

### Current Project Phase

**Phase 1:** ✅ COMPLETED
- Environment setup, data loading, EDA, baseline testing

**Phase 2:** ⏭️ IN PROGRESS
- Train all 6 models with 5 seeds each (30 total runs)
- ResNet-50: ✅ DONE (95.49%)
- CrossViT: ⏭️ Next
- DenseNet, EfficientNet, ViT, Swin: Pending

**Phase 3:** Pending
- Statistical validation (95% CI, hypothesis tests)
- Error analysis and ablation studies

**Phase 4:** Pending
- Thesis writing and Flask demo

---

## Important Notes

### Academic Integrity
- All experiments use `seed=42` for reproducibility
- Statistical validation required (95% CI, p-values)
- Code similarity <20% for Turnitin
- Follow TAR UMT FYP standards

### Data Privacy
- Do NOT commit raw dataset to GitHub (too large)
- Do NOT commit trained models to GitHub (too large)
- Use `.gitignore` to exclude large files
- Dataset is for academic use only (cite properly)

### Citation

**Dataset:**
```
Rahman, T., et al. (2021). COVID-19 Radiography Database.
Kaggle. https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
```

**CrossViT Model:**
```
Chen, C. F., Fan, Q., & Panda, R. (2021). CrossViT: Cross-Attention
Multi-Scale Vision Transformer for Image Classification.
ICCV 2021.
```

---

## Getting Help

### Documentation
- `CLAUDE.md` - Complete project instructions (READ THIS FIRST!)
- `WORKSTATION_STEP_BY_STEP.md` - Remote desktop guide
- `USING_CLAUDE_CODE_ON_WORKSTATION.md` - Claude Code usage

### Tools
- Use Claude Code for real-time assistance
- Check MLflow UI for experiment tracking
- Read notebook markdown cells for explanations

### Support
- Lab staff for hardware/network issues
- Supervisor for project questions
- GitHub issues for code problems

---

## License

This is an academic project for TAR UMT Final Year Project (FYP).
Code is available for educational purposes.

---

## Contact

**Student:** Tan Ming Kai
**Student ID:** 24PMR12003
**Institution:** TAR UMT
**Program:** Data Science

---

**Good luck with your training!** 🚀

Remember: **COMPLETION > PERFECTION** - The goal is to pass (50%+), not publication quality.
