# Guide to Create Remaining Baseline Notebooks (08-11)

**Status:** CrossViT training is running in background ✅
**Template:** Use `07_resnet50_training.ipynb` as base

---

## Quick Instructions

Copy `07_resnet50_training.ipynb` for each remaining model and make these specific changes:

---

## 08 - DenseNet-121 Training

**Copy command:**
```bash
cp notebooks/07_resnet50_training.ipynb notebooks/08_densenet121_training.ipynb
```

**Changes needed:**

### Cell 0 (Markdown - Title):
```markdown
# 08 - DenseNet-121 Training (Phase 2)

**Model:** DenseNet-121 (8M parameters, dense connections)
```

### Cell 1 (Imports):
No changes needed

### Cell 3 (CONFIG):
```python
CONFIG = {
    'device': device,
    'model_name': 'DenseNet-121',  # CHANGE
    'num_classes': 4,
    'image_size': 240,
    'class_names': ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'],
    'class_weights': [1.47, 0.52, 0.88, 3.95],
    'batch_size': 16,  # CHANGE (smaller than ResNet)
    'num_workers': 0,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'max_epochs': 30,
    'early_stopping_patience': 10,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'mixed_precision': True,
    'seeds': [42, 123, 456, 789, 101112],
}
```

### Cell 9 (Model Loading in train_single_seed function):
**Replace this section:**
```python
    # Load model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
    model = model.to(device)

    print(f"✅ ResNet-50 loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
```

**With:**
```python
    # Load model
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, config['num_classes'])
    model = model.to(device)

    print(f"✅ DenseNet-121 loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
```

**And update function name and all file names:**
- `train_resnet_single_seed` → `train_densenet_single_seed`
- `"resnet50-seed-{seed}"` → `"densenet121-seed-{seed}"`
- `f"resnet50_best_seed{seed}.pth"` → `f"densenet121_best_seed{seed}.pth"`
- `f"resnet50_cm_seed{seed}.png"` → `f"densenet121_cm_seed{seed}.png"`
- `"ResNet-50 Confusion Matrix"` → `"DenseNet-121 Confusion Matrix"`
- `"resnet50_results.csv"` → `"densenet121_results.csv"`
- `"ResNet-50 Results"` → `"DenseNet-121 Results"`

---

## 09 - EfficientNet-B0 Training

**Copy command:**
```bash
cp notebooks/07_resnet50_training.ipynb notebooks/09_efficientnet_training.ipynb
```

**Changes needed:**

### Cell 0 (Markdown):
```markdown
# 09 - EfficientNet-B0 Training (Phase 2)

**Model:** EfficientNet-B0 (5.3M parameters, compound scaling)
```

### Cell 1 (Imports - ADD timm):
```python
import timm  # ADD THIS LINE after torchvision imports
```

### Cell 3 (CONFIG):
```python
CONFIG = {
    'device': device,
    'model_name': 'EfficientNet-B0',  # CHANGE
    'num_classes': 4,
    'image_size': 240,
    'class_names': ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'],
    'class_weights': [1.47, 0.52, 0.88, 3.95],
    'batch_size': 16,  # CHANGE
    'num_workers': 0,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'max_epochs': 30,
    'early_stopping_patience': 10,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'mixed_precision': True,
    'seeds': [42, 123, 456, 789, 101112],
}
```

### Cell 9 (Model Loading):
**Replace:**
```python
    # Load model
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=config['num_classes'])
    model = model.to(device)

    print(f"✅ EfficientNet-B0 loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
```

**Update all names:**
- Function: `train_efficientnet_single_seed`
- MLflow run: `"efficientnet-seed-{seed}"`
- Files: `efficientnet_best_seed{seed}.pth`, `efficientnet_cm_seed{seed}.png`
- Display: "EfficientNet-B0"
- Results: `efficientnet_results.csv`

---

## 10 - ViT-Base/16 Training

**Copy command:**
```bash
cp notebooks/07_resnet50_training.ipynb notebooks/10_vit_training.ipynb
```

**Changes needed:**

### Cell 0:
```markdown
# 10 - ViT-Base/16 Training (Phase 2)

**Model:** ViT-Base/16 (86M parameters, pure transformer)
```

### Cell 1 (Imports):
```python
import timm  # ADD THIS
```

### Cell 3 (CONFIG):
```python
CONFIG = {
    'device': device,
    'model_name': 'ViT-Base/16',  # CHANGE
    'num_classes': 4,
    'image_size': 224,  # CHANGE (ViT uses 224)
    'class_names': ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'],
    'class_weights': [1.47, 0.52, 0.88, 3.95],
    'batch_size': 8,  # CHANGE (large model, smaller batch)
    'num_workers': 0,
    'learning_rate': 5e-5,  # CHANGE (transformer learning rate)
    'weight_decay': 1e-4,
    'max_epochs': 30,
    'early_stopping_patience': 10,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'mixed_precision': True,
    'seeds': [42, 123, 456, 789, 101112],
}
```

### Cell 7 (Transforms - UPDATE image_size to 224):
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # CHANGE to 224
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=CONFIG['mean'], std=CONFIG['std'])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # CHANGE to 224
    transforms.ToTensor(),
    transforms.Normalize(mean=CONFIG['mean'], std=CONFIG['std'])
])
```

### Cell 9 (Model Loading):
```python
    # Load model
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=config['num_classes'])
    model = model.to(device)

    print(f"✅ ViT-Base/16 loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
```

**Update all names:**
- Function: `train_vit_single_seed`
- MLflow: `"vit-seed-{seed}"`
- Files: `vit_best_seed{seed}.pth`, `vit_cm_seed{seed}.png`
- Display: "ViT-Base/16"
- Results: `vit_results.csv`

---

## 11 - Swin-Tiny Training

**Copy command:**
```bash
cp notebooks/07_resnet50_training.ipynb notebooks/11_swin_training.ipynb
```

**Changes needed:**

### Cell 0:
```markdown
# 11 - Swin-Tiny Training (Phase 2)

**Model:** Swin-Tiny (28M parameters, hierarchical transformer)
```

### Cell 1:
```python
import timm  # ADD THIS
```

### Cell 3 (CONFIG):
```python
CONFIG = {
    'device': device,
    'model_name': 'Swin-Tiny',  # CHANGE
    'num_classes': 4,
    'image_size': 224,  # CHANGE (Swin uses 224)
    'class_names': ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'],
    'class_weights': [1.47, 0.52, 0.88, 3.95],
    'batch_size': 12,  # CHANGE
    'num_workers': 0,
    'learning_rate': 5e-5,  # CHANGE
    'weight_decay': 1e-4,
    'max_epochs': 30,
    'early_stopping_patience': 10,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'mixed_precision': True,
    'seeds': [42, 123, 456, 789, 101112],
}
```

### Cell 7 (Transforms):
```python
# Change image_size to 224 in both transforms
transforms.Resize((224, 224))
```

### Cell 9 (Model Loading):
```python
    # Load model
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=config['num_classes'])
    model = model.to(device)

    print(f"✅ Swin-Tiny loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
```

**Update all names:**
- Function: `train_swin_single_seed`
- MLflow: `"swin-seed-{seed}"`
- Files: `swin_best_seed{seed}.pth`, `swin_cm_seed{seed}.png`
- Display: "Swin-Tiny"
- Results: `swin_results.csv`

---

## Quick Reference: Model Specifications

| Model | Batch | LR | Image Size | Library | Model Code |
|-------|-------|-------|------------|---------|------------|
| DenseNet-121 | 16 | 1e-4 | 240 | torchvision | `models.densenet121()` |
| EfficientNet-B0 | 16 | 1e-4 | 240 | timm | `timm.create_model('efficientnet_b0')` |
| ViT-Base/16 | 8 | 5e-5 | 224 | timm | `timm.create_model('vit_base_patch16_224')` |
| Swin-Tiny | 12 | 5e-5 | 224 | timm | `timm.create_model('swin_tiny_patch4_window7_224')` |

---

## Find & Replace Helper

For each notebook, use these find/replace patterns:

**DenseNet-121:**
- `resnet` → `densenet` (case-insensitive)
- `ResNet-50` → `DenseNet-121`
- `model.fc` → `model.classifier`

**EfficientNet-B0:**
- `resnet` → `efficientnet`
- `ResNet-50` → `EfficientNet-B0`

**ViT-Base/16:**
- `resnet` → `vit`
- `ResNet-50` → `ViT-Base/16`
- `240` → `224` (in transforms only)

**Swin-Tiny:**
- `resnet` → `swin`
- `ResNet-50` → `Swin-Tiny`
- `240` → `224` (in transforms only)

---

## ⚡ Fastest Approach

1. **Open** `07_resnet50_training.ipynb` in Jupyter
2. **"Save As"** with new name (e.g., `08_densenet121_training.ipynb`)
3. **Find & Replace All** (Ctrl+H):
   - Search: `resnet` → Replace: `densenet` (match case OFF)
   - Search: `ResNet-50` → Replace: `DenseNet-121` (match case ON)
4. **Update Cell 3** (CONFIG) with correct batch_size, lr, model_name
5. **Update Cell 9** (model loading code)
6. **Save** and repeat for other models

---

## Verification Checklist

For each notebook, verify:
- [ ] Title shows correct model name
- [ ] CONFIG has correct model_name, batch_size, learning_rate
- [ ] Model loading uses correct library (torchvision vs timm)
- [ ] All file names use correct prefix (densenet, efficientnet, vit, swin)
- [ ] MLflow run name is correct
- [ ] Image size is 224 for ViT and Swin (not 240)

---

**Status:** CrossViT (06) is training in background ✅
**Priority:** Create these 4 notebooks while CrossViT trains
**Time:** ~10-15 minutes per notebook (copy + modify)
