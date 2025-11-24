# Swin/ViT Image Size Fix - 2025-11-19

## Problem

Training Swin and ViT models failed with error:
```
AssertionError: Input height (240) doesn't match model (224).
```

## Root Cause

The `train_all_models_safe.py` script had hardcoded 240×240 image size for all models, but ViT and Swin transformers expect 224×224 images (as indicated by "224" in their model names).

### Specific Issues:

1. **Model loading** (lines 207, 210):
   ```python
   # WRONG - forces 240 on models that expect 224
   model = timm.create_model('vit_base_patch16_224', pretrained=True, img_size=240, num_classes=num_classes)
   model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, img_size=240, num_classes=num_classes)
   ```

2. **Dataset preprocessing** (line 98):
   ```python
   # WRONG - hardcoded 240 for all models
   image = cv2.resize(image, (240, 240))
   ```

## Solution Applied

### Change 1: Updated `get_model()` function (lines 207-212)

**Before:**
```python
elif model_name == 'vit':
    import timm
    model = timm.create_model('vit_base_patch16_224', pretrained=True, img_size=240, num_classes=num_classes)
elif model_name == 'swin':
    import timm
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, img_size=240, num_classes=num_classes)
```

**After:**
```python
elif model_name == 'vit':
    import timm
    # FIXED: Removed img_size=240 - ViT expects 224×224 by default
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
elif model_name == 'swin':
    import timm
    # FIXED: Removed img_size=240 - Swin expects 224×224 by default
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
```

### Change 2: Updated `COVID19DatasetRaw` class (lines 74, 99)

**Before:**
```python
class COVID19DatasetRaw(Dataset):
    def __init__(self, dataframe, transform=None, apply_clahe=True):
        # ...

    def __getitem__(self, idx):
        # ...
        image = cv2.resize(image, (240, 240))  # Hardcoded 240
```

**After:**
```python
class COVID19DatasetRaw(Dataset):
    def __init__(self, dataframe, transform=None, apply_clahe=True, image_size=240):
        self.image_size = image_size  # FIXED: Configurable image size
        # ...

    def __getitem__(self, idx):
        # ...
        image = cv2.resize(image, (self.image_size, self.image_size))  # Uses configurable size
```

### Change 3: Updated `train_model_single_seed()` function (lines 228-236)

**Before:**
```python
def train_model_single_seed(model_name, seed, config, train_df, val_df, test_df):
    set_seed(seed)

    # Create datasets
    train_dataset = COVID19DatasetRaw(train_df, transform=train_transform)
    val_dataset = COVID19DatasetRaw(val_df, transform=val_transform)
    test_dataset = COVID19DatasetRaw(test_df, transform=val_transform)
```

**After:**
```python
def train_model_single_seed(model_name, seed, config, train_df, val_df, test_df):
    set_seed(seed)

    # Determine image size based on model
    # ViT and Swin expect 224×224, others use 240×240
    image_size = 224 if model_name in ['vit', 'swin'] else config['image_size']
    print(f"Using image size: {image_size}×{image_size}")

    # Create datasets
    train_dataset = COVID19DatasetRaw(train_df, transform=train_transform, image_size=image_size)
    val_dataset = COVID19DatasetRaw(val_df, transform=val_transform, image_size=image_size)
    test_dataset = COVID19DatasetRaw(test_df, transform=val_transform, image_size=image_size)
```

### Change 4: Updated BASE_CONFIG documentation (line 56)

```python
'image_size': 240,  # Default for ResNet, DenseNet, EfficientNet, CrossViT (ViT/Swin use 224)
```

## Image Size Summary

| Model | Image Size | Notes |
|-------|------------|-------|
| ResNet-50 | 240×240 | Default config |
| DenseNet-121 | 240×240 | Default config |
| EfficientNet-B0 | 240×240 | Default config |
| CrossViT-Tiny | 240×240 | Model name indicates 240 |
| **ViT-Base** | **224×224** | Model name indicates 224 (fixed) |
| **Swin-Tiny** | **224×224** | Model name indicates 224 (fixed) |

## Verification

```bash
# Syntax check passed
python -m py_compile train_all_models_safe.py
# No errors
```

## Next Steps

1. Test Swin with single seed to verify fix:
   ```bash
   # Will train just seed 42 to test
   python train_all_models_safe.py swin
   ```

2. If successful, complete all Swin training (5 seeds):
   ```bash
   python train_all_models_safe.py swin
   ```

3. Complete ViT training (3 remaining seeds):
   ```bash
   python train_all_models_safe.py vit
   ```

## Expected Outcome

- Swin and ViT will now receive correctly sized 224×224 images
- No more "Input height mismatch" errors
- Training should proceed normally like other models

## Impact

- **Fixes:** 5 blocked Swin experiments + any future ViT/Swin runs
- **No impact on:** Already completed models (ResNet, DenseNet, EfficientNet, CrossViT) - they continue using 240×240
- **Backward compatible:** Default image_size=240 maintains behavior for non-ViT/Swin models

---

**Fixed by:** Claude Code
**Date:** 2025-11-19
**Files modified:** `train_all_models_safe.py`
