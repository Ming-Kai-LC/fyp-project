# Phase 4: Documentation & Deployment

**Duration:** Week 9-10 (Upcoming)
**Goal:** Thesis writing and Flask demo deployment
**Status:** ⏸️ PENDING (Waiting for Phase 3 completion)

---

## 📋 Phase Overview

Phase 4 focuses on documenting all research findings in the thesis (especially Chapters 4 & 5) and creating a functional Flask web application for model deployment demonstration.

---

## 🎯 Objectives

### 1. Thesis Writing ⏸️
- Complete Chapter 4 (Methodology & Experimental Setup)
- Complete Chapter 5 (Results & Discussion)
- Generate reproducibility statement
- Format all tables and figures

### 2. Flask Demo ⏸️
- Create basic web interface
- Implement model inference pipeline
- Add file upload functionality
- Display predictions with confidence scores

### 3. Documentation ⏸️
- Code documentation
- README files
- User guide for Flask demo
- Deployment instructions

---

## 📊 Planned Notebooks

### 15_thesis_content.ipynb ⏸️

**Purpose:** Generate all content for thesis Chapters 4 & 5

**Chapter 4 Content (Methodology):**

**4.1 Experimental Setup**
- [ ] Hardware specifications table
- [ ] Software environment table
- [ ] Dataset statistics summary
- [ ] Training configuration parameters

**4.2 Data Preprocessing**
- [ ] CLAHE parameters and justification
- [ ] Image size selection rationale
- [ ] Data split methodology
- [ ] Class weight calculation

**4.3 Model Architectures**
- [ ] Brief description of each model
- [ ] Parameter counts comparison
- [ ] Architecture diagrams (if needed)
- [ ] Pre-training details

**4.4 Training Protocol**
- [ ] Hyperparameters (with justification)
- [ ] Optimization algorithm
- [ ] Early stopping criteria
- [ ] Batch size selection
- [ ] Data augmentation strategy

**4.5 Evaluation Metrics**
- [ ] Accuracy, Precision, Recall, F1-score
- [ ] Confusion matrices
- [ ] Statistical validation methods
- [ ] Cross-validation strategy (5 seeds)

**4.6 Reproducibility Statement**
```
All experiments were conducted with fixed random seeds (42, 123, 456, 789, 101112)
to ensure reproducibility. The complete experimental setup is detailed below:

Hardware:
- GPU: NVIDIA RTX 6000 Ada Generation (48 GB VRAM)
- CPU: [Specify]
- RAM: [Specify]
- OS: Windows 11

Software:
- Python: 3.13
- PyTorch: 2.x
- CUDA: 13.0
- Key libraries: timm, torchvision, opencv-python

Training Configuration:
- Batch size: 380 (auto-adjusted to 323 for single user)
- Learning rate: 1.78e-4
- Optimizer: Adam (weight_decay=1e-4)
- Mixed precision: Enabled
- Random seeds: [42, 123, 456, 789, 101112]

All code and trained models are available at: [GitHub URL]
```

---

**Chapter 5 Content (Results & Discussion):**

**5.1 Model Performance Summary**
- [ ] Table 5.1: Overall performance (all models)
- [ ] Figure 5.1: Performance comparison (boxplot)
- [ ] Figure 5.2: Confidence intervals visualization
- [ ] Discussion of ranking

**5.2 Statistical Validation**
- [ ] Table 5.2: Hypothesis testing results
- [ ] Table 5.3: Confidence intervals
- [ ] Table 5.4: Effect sizes
- [ ] Discussion of statistical significance

**5.3 Per-Class Analysis**
- [ ] Table 5.5: Per-class metrics (all models)
- [ ] Figure 5.3: Per-class performance radar chart
- [ ] Figure 5.4: Aggregated confusion matrices
- [ ] Discussion of class-specific performance

**5.4 Error Analysis**
- [ ] Figure 5.5: Error distribution across models
- [ ] Figure 5.6: Common misclassification patterns
- [ ] Table 5.6: Most confused class pairs
- [ ] Discussion of failure modes

**5.5 Ablation Studies**
- [ ] Table 5.7: CLAHE impact (H₂)
- [ ] Table 5.8: Augmentation impact (H₃)
- [ ] Table 5.9: Dual-branch impact (H₄)
- [ ] Discussion of design choices

**5.6 Discussion**

**5.6.1 Why CNNs Outperform CrossViT**
- Inductive bias and translation invariance
- Local feature extraction superiority
- Data efficiency considerations
- Pre-training domain mismatch
- Medical imaging specific patterns

**5.6.2 Practical Implications**
- DenseNet-121 recommended for deployment (most consistent)
- Model size vs performance trade-off
- Computational efficiency considerations

**5.6.3 Limitations**
- Single dataset (COVID-19 Radiography Database)
- Class imbalance despite weighting
- Limited to 4-class classification
- Transformer models may need larger datasets

**5.6.4 Future Work**
- Multi-dataset validation
- Ensemble methods
- Attention visualization
- Clinical validation study
- Larger transformer pre-training

---

### 16_flask_demo.ipynb ⏸️

**Purpose:** Develop and test Flask web application

**Flask App Features:**

**1. Model Loading**
```python
import torch
from flask import Flask, render_template, request
import timm

app = Flask(__name__)

# Load best model (DenseNet-121 seed 789)
model = load_model('densenet121_best_seed789.pth')
model.eval()
```

**2. Image Upload & Preprocessing**
```python
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    image = preprocess_image(file)  # CLAHE + resize + normalize
    return image
```

**3. Prediction Pipeline**
```python
def predict(image):
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

    return {
        'class': class_names[predicted_class],
        'confidence': probabilities[predicted_class].item(),
        'all_probabilities': probabilities[0].tolist()
    }
```

**4. Web Interface**
```html
<!-- templates/index.html -->
<h1>COVID-19 X-Ray Classification</h1>
<form method="POST" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*">
    <button type="submit">Classify</button>
</form>

<!-- Results -->
<div class="results">
    <h2>Prediction: {{ prediction.class }}</h2>
    <p>Confidence: {{ prediction.confidence }}%</p>
    <canvas id="probChart"></canvas>  <!-- Probability bar chart -->
</div>
```

**5. Visualization**
- Display uploaded image
- Show prediction with confidence score
- Bar chart of all class probabilities
- Grad-CAM heatmap (optional)

---

## 📁 Expected Files Generated in Phase 4

### Thesis Content:
```
thesis/
├── chapter4_methodology.tex
├── chapter5_results.tex
├── tables/
│   ├── table_4_1_hardware.tex
│   ├── table_4_2_dataset_stats.tex
│   ├── table_5_1_performance.tex
│   ├── table_5_2_hypothesis_tests.tex
│   └── ... (all other tables)
└── figures/
    ├── fig_4_1_preprocessing_pipeline.pdf
    ├── fig_5_1_performance_comparison.pdf
    ├── fig_5_2_confidence_intervals.pdf
    └── ... (all other figures)
```

### Flask App:
```
flask_demo/
├── app.py                    # Main Flask application
├── models/
│   └── densenet121_best_seed789.pth
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── app.js
│   └── uploads/             # Temporary uploaded images
├── templates/
│   ├── index.html
│   └── results.html
├── utils/
│   ├── preprocessing.py     # CLAHE + transforms
│   ├── model_loader.py
│   └── inference.py
├── requirements.txt
└── README.md
```

### Documentation:
```
docs/
├── README.md                # Project overview
├── INSTALLATION.md          # Setup instructions
├── USAGE.md                 # How to run experiments
├── FLASK_DEPLOYMENT.md      # Flask app deployment
└── API_REFERENCE.md         # Code documentation
```

---

## 🌐 Flask Demo Requirements

### Minimum Viable Product (MVP):

**Must Have:**
- [x] File upload functionality
- [x] Image preprocessing (CLAHE)
- [x] Model inference
- [x] Display prediction + confidence
- [x] Basic CSS styling

**Nice to Have:**
- [ ] Multiple model selection
- [ ] Batch upload
- [ ] Grad-CAM visualization
- [ ] Result history
- [ ] Export results to PDF

**Not Required (Out of Scope):**
- ❌ User authentication
- ❌ Database integration
- ❌ RESTful API
- ❌ Mobile app
- ❌ Production deployment

---

## 📊 Thesis Tables & Figures Checklist

### Chapter 4 (Methodology):

**Tables:**
- [ ] Table 4.1: Hardware & Software Specifications
- [ ] Table 4.2: Dataset Statistics (train/val/test split)
- [ ] Table 4.3: Class Distribution & Weights
- [ ] Table 4.4: Training Hyperparameters
- [ ] Table 4.5: Model Architectures Comparison

**Figures:**
- [ ] Figure 4.1: Preprocessing Pipeline Diagram
- [ ] Figure 4.2: Sample Images (before/after CLAHE)
- [ ] Figure 4.3: Data Augmentation Examples
- [ ] Figure 4.4: Training Workflow Diagram

---

### Chapter 5 (Results):

**Tables:**
- [ ] Table 5.1: Model Performance Summary (Mean ± Std, CI)
- [ ] Table 5.2: Statistical Comparison (McNemar's test)
- [ ] Table 5.3: Per-Class Metrics (Precision, Recall, F1)
- [ ] Table 5.4: Confusion Matrix Summary
- [ ] Table 5.5: Ablation Study Results

**Figures:**
- [ ] Figure 5.1: Model Performance Boxplot
- [ ] Figure 5.2: Confidence Intervals Visualization
- [ ] Figure 5.3: Per-Class Performance Radar Chart
- [ ] Figure 5.4: Aggregated Confusion Matrices
- [ ] Figure 5.5: Error Analysis Heatmap
- [ ] Figure 5.6: Training Curves (best models)
- [ ] Figure 5.7: CLAHE Comparison (ablation)
- [ ] Figure 5.8: Augmentation Impact (ablation)

---

## ✅ Phase 4 Success Criteria

**Thesis:**
- [ ] Chapter 4 complete (15-20 pages)
- [ ] Chapter 5 complete (20-25 pages)
- [ ] All tables formatted (APA 7th edition)
- [ ] All figures high-resolution (300 DPI)
- [ ] Reproducibility statement included
- [ ] References properly cited

**Flask Demo:**
- [ ] App runs locally
- [ ] Can upload and classify images
- [ ] Predictions displayed correctly
- [ ] Basic UI is functional and styled
- [ ] README with deployment instructions

**Documentation:**
- [ ] All notebooks have markdown explanations
- [ ] Code comments added
- [ ] README files for each directory
- [ ] Installation guide complete

---

## ⏱️ Estimated Timeline

**Week 9:**
- Day 1-2: Generate thesis tables & figures (Notebook 15)
- Day 3-4: Write Chapter 4 (Methodology)
- Day 5-6: Write Chapter 5 intro & results sections
- Day 7: Write Chapter 5 discussion section

**Week 10:**
- Day 1-2: Develop Flask app (Notebook 16)
- Day 3: Test Flask app locally
- Day 4: Write documentation
- Day 5: Final review and polish
- Day 6-7: Buffer for unexpected issues

**Total Time:** ~2 weeks

---

## 🚀 Deployment Instructions (Flask App)

### Local Deployment:

```bash
# 1. Navigate to flask_demo directory
cd flask_demo

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Flask app
python app.py

# 5. Open browser
# Navigate to: http://localhost:5000
```

### Requirements.txt:
```
Flask==2.3.0
torch==2.0.0
torchvision==0.15.0
opencv-python==4.8.0
Pillow==10.0.0
numpy==1.24.0
```

---

## 📚 References for Phase 4

- American Psychological Association (2020) - Publication Manual (7th ed.)
- Flask Documentation - https://flask.palletsprojects.com/
- Selvaraju et al. (2017) - Grad-CAM: Visual Explanations from Deep Networks
- TAR UMT FYP Handbook - Thesis formatting requirements

---

## 📝 Thesis Chapter Templates

### Chapter 4 Structure:

```
4. METHODOLOGY
   4.1 Experimental Setup
       4.1.1 Hardware & Software Environment
       4.1.2 Dataset Description
       4.1.3 Evaluation Protocol
   4.2 Data Preprocessing
       4.2.1 CLAHE Enhancement
       4.2.2 Image Resizing & Normalization
       4.2.3 Data Augmentation
   4.3 Model Architectures
       4.3.1 CNN Baselines
       4.3.2 Transformer Baselines
       4.3.3 CrossViT (Proposed Model)
   4.4 Training Configuration
       4.4.1 Hyperparameters
       4.4.2 Optimization Strategy
       4.4.3 Regularization Techniques
   4.5 Statistical Validation
       4.5.1 Cross-Validation Strategy
       4.5.2 Confidence Intervals
       4.5.3 Hypothesis Testing
   4.6 Reproducibility Statement
```

### Chapter 5 Structure:

```
5. RESULTS AND DISCUSSION
   5.1 Model Performance Summary
       5.1.1 Overall Accuracy
       5.1.2 Confidence Intervals
       5.1.3 Model Ranking
   5.2 Statistical Analysis
       5.2.1 Hypothesis Testing Results
       5.2.2 Effect Sizes
       5.2.3 Significance of Differences
   5.3 Per-Class Performance
       5.3.1 Precision, Recall, F1-Score
       5.3.2 Confusion Matrix Analysis
       5.3.3 Class-Specific Observations
   5.4 Error Analysis
       5.4.1 Misclassification Patterns
       5.4.2 Failure Modes
       5.4.3 Error Distribution
   5.5 Ablation Studies
       5.5.1 CLAHE Impact
       5.5.2 Data Augmentation Impact
       5.5.3 CrossViT Dual-Branch Analysis
   5.6 Discussion
       5.6.1 Hypothesis Validation
       5.6.2 CNN vs Transformer Performance
       5.6.3 Practical Implications
       5.6.4 Limitations
       5.6.5 Future Work
```

---

**Phase 4 Start Date:** TBD (After Phase 3 completes)
**Phase 4 Status:** ⏸️ PENDING
**Prerequisites:** Phase 3 statistical analysis complete
**Final Deliverable:** Complete thesis Chapters 4-5 + Flask demo
