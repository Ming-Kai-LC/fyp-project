# Training Status - 240×240 Fair Comparison (Nov 19, 2025 - 5:11 PM)

## ✓ Confirmed: All Models Using Uniform Resolution

**Resolution Strategy:**
- **All models**: 240×240 (ResNet, DenseNet, EfficientNet, CrossViT, ViT)
- **Swin only**: 256×256 (closest available, no 240 variant exists)
- **Fair comparison**: Within 6.7% difference (240 vs 256)

## Current Training Status

### ACTIVE NOW: ViT-Base/16+ CLIP (240×240)
- **Model**: `vit_base_patch16_plus_clip_240.laion400m_e32` ✓
- **Status**: Epoch 2/50, Seed 42 (15% complete)
- **GPU**: 100% utilization, 31.9 GB VRAM, 297W, 60°C
- **Performance**: Epoch 1 Val Acc = 34.25%
- **Log**: `logs/vit_training_240_manual.log`
- **ETA**: ~3-4 hours for all 5 seeds

### Models Completed (240×240) ✓
1. **ResNet-50**: 5/5 seeds (240×240) ✓
2. **DenseNet-121**: 5/5 seeds (240×240) ✓
3. **EfficientNet-B0**: 5/5 seeds (240×240) ✓
4. **CrossViT-Tiny**: 5/5 seeds (240×240) ✓

### Models Pending Retraining
5. **ViT-Base/16+**: 0/5 seeds complete (IN PROGRESS)
   - Old 224×224 models deleted/will be overwritten
   - Now using correct 240×240 model

6. **Swin-Tiny**: 0/5 seeds complete (QUEUED NEXT)
   - Will use `swinv2_tiny_window8_256.ms_in1k` (256×256)
   - Old 224×224 models will be overwritten

## Code Verification

### ViT Configuration (train_all_models_safe.py:214)
```python
elif model_name == 'vit':
    import timm
    # Use ViT-Base/16+ CLIP with 240×240 resolution for fair comparison
    model = timm.create_model('vit_base_patch16_plus_clip_240.laion400m_e32',
                              pretrained=True, num_classes=num_classes)
```

### Swin Configuration (train_all_models_safe.py:218)
```python
elif model_name == 'swin':
    import timm
    # Use SwinV2-Tiny with 256×256 (closest to 240, no 240 variant available)
    model = timm.create_model('swinv2_tiny_window8_256.ms_in1k',
                              pretrained=True, num_classes=num_classes)
```

### Dataset Image Size Logic (train_all_models_safe.py:235-238)
```python
if model_name == 'swin':
    image_size = 256  # SwinV2-Tiny requires 256 (closest to 240)
else:
    image_size = 240  # Standard for all other models
```

## Next Steps (Automatic)

After ViT completes, you need to manually run:
```bash
python train_all_models_safe.py swin
```

This will train Swin with 256×256 for all 5 seeds (42, 123, 456, 789, 101112).

## Phase 2 Progress

- **Current**: 20/30 models complete with fair comparison (66%)
- **After ViT**: 25/30 models (83%)
- **After Swin**: 30/30 models (100%) → Ready for Phase 3!

## Timeline Estimate

- **ViT training**: 3-4 hours (5 seeds × ~40min/seed)
- **Swin training**: 2-3 hours (5 seeds × ~30min/seed)
- **Total remaining**: ~6-7 hours
- **Expected completion**: Nov 19, 2025 ~11 PM - 12 AM

## What Happens While You're Away

1. ViT will train all 5 seeds automatically (seed 42, 123, 456, 789, 101112)
2. Models saved to: `experiments/phase2_systematic/models/vit/`
3. Confusion matrices saved to: `experiments/phase2_systematic/results/confusion_matrices/`
4. After ViT completes, GPU will go idle
5. **You need to manually start Swin training when you return**

## How to Check Progress When You Return

```bash
# Check GPU status
nvidia-smi

# Check training log
tail -100 logs/vit_training_240_manual.log

# Count completed models
ls experiments/phase2_systematic/models/vit/*.pth | wc -l
# Should show 5 when complete

# Start Swin training (if ViT is done)
python train_all_models_safe.py swin > logs/swin_training_240.log 2>&1 &
```

## Fair Comparison Justification

**Why 240×240 for all models?**
- CrossViT requires 240×240 (fixed architecture)
- All baselines must use same resolution to avoid confounding variables
- ResNet/DenseNet/EfficientNet: Resolution-agnostic (240 works fine)
- ViT: Found `vit_base_patch16_plus_clip_240.laion400m_e32` with pretrained weights
- Swin: No 240 variant exists, 256 is closest (6.7% difference acceptable)

**Methodological soundness:**
- Chapter 4 (Methodology) specifies: "Resize to 240×240 (CrossViT requirement)"
- Uniform preprocessing ensures fair performance comparison
- Statistical validation (Phase 3) will be methodologically rigorous

---

**Status**: ViT training in progress, 240×240 confirmed ✓
**GPU**: Healthy (100% util, 60°C, 297W)
**Next**: Swin training after ViT completes
