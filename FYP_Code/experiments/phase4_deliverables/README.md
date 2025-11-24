# Phase 4: Documentation & Deployment

**Timeline:** Weeks 9-10
**Status:** ⏭️ Ready to start

---

## Objectives

1. Generate all thesis content (tables, figures)
2. Create reproducibility statement for Chapter 4
3. Build basic Flask demo for model deployment

---

## Folder Structure

```
phase4_deliverables/
├── thesis_content/
│   ├── chapter4_tables/      # Reproducibility tables
│   └── chapter5_figures/     # Results figures with captions
└── flask_demo/
    ├── app.py                # Flask application
    ├── templates/            # HTML templates
    │   └── index.html
    └── static/               # CSS, JS, images
        ├── css/
        └── js/
```

---

## Deliverables

### 1. Thesis Content (Notebook 15)

**Chapter 4 Tables:**
- Hyperparameter configuration table
- Training environment specifications
- Data split statistics
- Reproducibility checklist

**Chapter 5 Figures:**
- Model performance comparison (with CIs)
- Confusion matrices
- Per-class F1-score comparison
- Medical metrics visualization
- All figures with captions

**Format:** LaTeX tables + high-res PNG figures (300 DPI)

---

### 2. Flask Demo (Notebook 16)

**Features:**
- Upload chest X-ray image
- Preprocess image (CLAHE, resize)
- Run inference with best model (ResNet-50 or CrossViT)
- Display prediction with confidence scores
- Show class probabilities

**Tech Stack:**
- Backend: Flask (Python)
- Frontend: HTML + Bootstrap CSS
- Model: PyTorch (.pth file)

**Requirements:**
- Basic functionality over aesthetics
- ~5 minute inference time acceptable
- No need for deployment (local demo only)

---

## Time Estimate

**Notebook 15 (Thesis Content):** 1-2 hours
- Generate tables from Phase 3 results
- Format figures with captions
- Export to LaTeX/Word format

**Notebook 16 (Flask Demo):** 2-3 hours
- Create Flask app skeleton
- Load trained model
- Implement inference pipeline
- Basic HTML interface

**Total:** 3-5 hours

---

## Success Criteria

### Notebook 15:
- [ ] All Chapter 4 tables generated
- [ ] All Chapter 5 figures ready (300 DPI)
- [ ] Reproducibility statement complete
- [ ] Files ready to copy into thesis

### Notebook 16:
- [ ] Flask app runs locally
- [ ] Can upload image
- [ ] Returns prediction correctly
- [ ] Shows confidence scores
- [ ] Basic error handling

---

## Next Steps

1. Run `notebooks/15_thesis_content.ipynb` (when ready to write thesis)
2. Run `notebooks/16_flask_demo.ipynb` (for FYP demonstration)
3. Test Flask app with sample images
4. Prepare for final submission

---

**Status:** Phase 3 complete ✅ | Phase 4 ready to start ⏭️
